import os
import sys
import re
import warnings
import urllib.parse

# ─── Fix Windows R Path for rpy2 ─────────────────────────────────────
if sys.platform == "win32":
    os.environ["R_HOME"] = r"C:\Program Files\R\R-4.6.1"
    r_bin_path = os.path.join(os.environ["R_HOME"], "bin", "x64")
    if r_bin_path not in os.environ["PATH"]:
        os.environ["PATH"] = r_bin_path + os.path.pathsep + os.environ["PATH"]
    try:
        os.add_dll_directory(r_bin_path)
    except AttributeError:
        pass

# ─── Standard Imports ───────────────────────────────────────────────
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

# Suppress Deprecation/RRuntime warnings before importing rpy2 subcomponents
warnings.filterwarnings("ignore", category=DeprecationWarning)
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# ─── Silence Windows 'sh' Console Subprocess Warnings ───────────────
import rpy2.rinterface_lib.callbacks
rpy2.rinterface_lib.callbacks.showmessage = lambda x: None

r = robjects.r
r.library("randomForest")

# ─── File Path Strategy ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "r_models")
DATA_DIR = os.path.join(BASE_DIR, "data")
TEMPLATE_DIR = os.path.join(ROOT_DIR, "frontend", "templates")
STATIC_DIR = os.path.join(ROOT_DIR, "frontend", "static")

# ─── Load Models & Datasets at Startup ──────────────────────────────
rf_model_path = os.path.join(MODELS_DIR, "mood_rf.rds")
km_path = os.path.join(MODELS_DIR, "kmeans_songs.rds")

r.assign("rf_model", r.readRDS(rf_model_path.replace("\\", "/")))
r.assign("km_path", km_path.replace("\\", "/"))
r("km_bundle <- readRDS(km_path)")

with localconverter(robjects.default_converter + pandas2ri.converter):
    km_center = np.array(r("km_bundle$center"))
    km_scale = np.array(r("km_bundle$scale"))
    km_centers = np.array(r("km_bundle$model$centers"))

# Load dataset
song_csv = os.path.join(DATA_DIR, "song_data.csv")
if not os.path.exists(song_csv):
    song_csv = os.path.join(DATA_DIR, "song_features.csv")

df_songs = pd.read_csv(song_csv)

# Dynamic Column Normalizer: Auto-detect variations like 'valence' vs 'audio_valence'
column_mapping = {}
for col in df_songs.columns:
    col_lower = col.lower().strip()
    if col_lower in ['valence', 'audio_valence', 'audio_valencies']:
        column_mapping[col] = 'audio_valence'
    elif col_lower in ['energy', 'audio_energy']:
        column_mapping[col] = 'energy'
    elif col_lower in ['danceability', 'danceable']:
        column_mapping[col] = 'danceability'
    elif col_lower in ['tempo', 'bpm', 'speed']:
        column_mapping[col] = 'tempo'
    elif col_lower in ['acousticness', 'acoustic']:
        column_mapping[col] = 'acousticness'
    elif col_lower in ['song_name', 'title', 'track_name', 'name']:
        column_mapping[col] = 'song_name'
    elif col_lower in ['duration_min', 'duration', 'length', 'duration_ms', 'song_duration_ms']:
        column_mapping[col] = 'duration_min'
    elif col_lower in ['song_popularity', 'popularity', 'track_popularity', 'pop', 'popularity_score']:
        column_mapping[col] = 'song_popularity'

df_songs = df_songs.rename(columns=column_mapping)

# Ensure duration in minutes
if 'duration_min' in df_songs.columns and df_songs['duration_min'].max() > 1000:
    df_songs['duration_min'] = df_songs['duration_min'] / 60000.0
elif 'duration_min' not in df_songs.columns:
    df_songs['duration_min'] = 3.5

# If popularity is missing from CSV, create default baseline
if 'song_popularity' not in df_songs.columns:
    df_songs['song_popularity'] = 50.0

if 'song_name' not in df_songs.columns:
    df_songs['song_name'] = [f"Track #{i+1}" for i in range(len(df_songs))]

# Ensure base clustering features exist in dataset
base_kmeans_features = ["audio_valence", "energy", "danceability", "tempo", "acousticness"]
for feat in base_kmeans_features:
    if feat not in df_songs.columns:
        df_songs[feat] = 0.5 if feat != 'tempo' else 110.0

# Mirror valence if needed
if 'valence' not in df_songs.columns and 'audio_valence' in df_songs.columns:
    df_songs['valence'] = df_songs['audio_valence']

# Pre-calculate cluster assignment once at startup for high performance
if 'cluster' not in df_songs.columns:
    feat_matrix = df_songs[base_kmeans_features].to_numpy(dtype=float)
    scaled_feats = (feat_matrix - km_center) / km_scale
    dists = np.linalg.norm(scaled_feats[:, np.newaxis, :] - km_centers[np.newaxis, :, :], axis=2)
    df_songs['cluster'] = np.argmin(dists, axis=1) + 1

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)


def analyze_text_emotion(text: str) -> str:
    """Parses text blocks to identify single-word core user emotion."""
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)

    lexicons = {
        "happy": ["good", "great", "happy", "love", "smile", "fun", "excited", "awesome", "joy", "glad", "cheerful", "delighted", "content", "wonderful", "peaceful", "blessed", "vibrant", "laugh", "bright"],
        "sad": ["sad", "cry", "lonely", "hurt", "bad", "down", "miss", "sorry", "gloomy", "heartbroke", "depressed", "unhappy", "sorrow", "grief", "miserable", "tears", "blue", "hopeless"],
        "stressed": ["busy", "work", "deadline", "stressed", "pressure", "overwhelmed", "exhausted", "tired", "burned", "fatigue", "overworked", "hectic", "drained"],
        "anxious": ["worry", "scared", "nervous", "anxious", "fear", "panic", "uncertain", "afraid", "tension", "uneasy", "fret", "restless", "dread"],
        "angry": ["mad", "angry", "hate", "annoyed", "pissed", "wrong", "furious", "rage", "irritated", "frustrated", "agitated", "bitter", "resentful"]
    }

    scores = {mood: sum(words.count(w) for w in words_list) for mood, words_list in lexicons.items()}
    max_mood = max(scores, key=scores.get)
    return max_mood if scores[max_mood] > 0 else "happy"


def get_mood_cluster_id(mood: str) -> int:
    """Finds the mathematical K-Means cluster index assigned to a mood archetype."""
    mood_centroids = {
        "happy": [0.8, 0.7, 0.8, 130, 0.1],
        "sad": [0.2, 0.2, 0.4, 70, 0.8],
        "stressed": [0.6, 0.8, 0.7, 140, 0.1],
        "anxious": [0.4, 0.3, 0.5, 80, 0.6],
        "angry": [0.7, 0.9, 0.8, 145, 0.05]
    }
    raw = np.array(mood_centroids.get(mood, [0.5, 0.5, 0.5, 100, 0.5]))
    scaled = (raw - km_center) / km_scale
    distances = np.linalg.norm(km_centers - scaled.reshape(1, -1), axis=1)
    return int(np.argmin(distances)) + 1


def predict_song_scores(cluster_id: int, mood: str = "happy") -> list:
    """Uses Random Forest model to rank matching songs within the chosen cluster."""
    candidates = df_songs[df_songs["cluster"] == cluster_id].copy()
    if candidates.empty:
        candidates = df_songs.copy()

    try:
        with localconverter(robjects.default_converter + pandas2ri.converter):
            model_terms = list(r("attr(terms(rf_model), 'term.labels')"))

        r_input = candidates.copy()
        for term in model_terms:
            if term not in r_input.columns:
                if term == "mood":
                    r_input["mood"] = mood
                elif term == "intensity":
                    intensity_map = {
                        "happy": "Medium",
                        "sad": "Low",
                        "stressed": "High",
                        "anxious": "Low",
                        "angry": "High"
                    }
                    r_input["intensity"] = intensity_map.get(mood, "Medium")
                elif term == "cluster":
                    r_input["cluster"] = int(cluster_id)
                elif term == "duration_min":
                    r_input["duration_min"] = 3.5
                else:
                    r_input[term] = 0.0

        safe_cols = [c for c in model_terms if c in r_input.columns]

        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_df = pandas2ri.py2rpy(r_input[safe_cols])
            r.assign("temp_df", r_df)
            r("""
            for (col in colnames(temp_df)) {
                levs <- rf_model$forest$xlevels[[col]]
                if (is.character(levs)) {
                    temp_df[[col]] <- factor(temp_df[[col]], levels = levs)
                } else {
                    temp_df[[col]] <- as.numeric(temp_df[[col]])
                }
            }
            preds <- predict(rf_model, newdata = temp_df)
            """)
            scores = np.array(r["preds"])
            candidates["predicted_score"] = scores
    except Exception as e:
        print(f"Notice: Using popularity fallback due to RF model predictor structure: {e}")
        if "song_popularity" in candidates.columns:
            candidates["predicted_score"] = candidates["song_popularity"].astype(float)
        else:
            candidates["predicted_score"] = (candidates["audio_valence"] * 50) + (candidates["energy"] * 50)

    # Handle missing scores
    if "song_popularity" in candidates.columns:
        candidates["predicted_score"] = candidates["predicted_score"].fillna(candidates["song_popularity"])
    candidates["predicted_score"] = candidates["predicted_score"].fillna(50.0)

    # Sort primarily by predicted_score, secondarily by song_popularity
    sort_cols = ["predicted_score"]
    if "song_popularity" in candidates.columns:
        sort_cols.append("song_popularity")
    candidates = candidates.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

    ranked_songs = []
    for i, row in candidates.head(5).iterrows():
        display_tempo = int(round(row["tempo"])) if "tempo" in row and pd.notna(row["tempo"]) else 120
        display_energy = round(float(row["energy"]), 2) if "energy" in row and pd.notna(row["energy"]) else 0.5
        display_title = str(row["song_name"]) if "song_name" in row and pd.notna(row["song_name"]) else f"Track #{i+1}"

        raw_score = row["predicted_score"]
        if pd.isna(raw_score) or np.isnan(raw_score):
            display_match = float(round(98.0 - (i * 4), 1))
        else:
            raw_score = float(raw_score)
            display_match = float(round(min(raw_score, 100.0) if raw_score > 1.0 else raw_score * 100.0, 1))

        # Generate Spotify Search Query URL
        encoded_title = urllib.parse.quote_plus(display_title)
        spotify_search_url = f"https://open.spotify.com/search/{encoded_title}"

        ranked_songs.append({
            "rank": i + 1,
            "title": display_title,
            "artist": "Rock / Alternative Track",
            "tempo": display_tempo,
            "energy": display_energy,
            "predicted_score": display_match,
            "track_url": spotify_search_url
        })

    return ranked_songs


# ─── App Routes ─────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    songs_list = None
    predicted_emotion = ""
    user_input = ""
    cluster_id = None

    if request.method == "POST":
        user_input = request.form.get("diary_entry", "").strip()
        if user_input:
            predicted_emotion = analyze_text_emotion(user_input)
            cluster_id = get_mood_cluster_id(predicted_emotion)
            songs_list = predict_song_scores(cluster_id, mood=predicted_emotion)

    return render_template(
        "index.html",
        songs=songs_list,
        user_input=user_input,
        predicted_emotion=predicted_emotion,
        cluster_id=cluster_id
    )


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """JSON API endpoint."""
    data = request.get_json(silent=True) or {}
    user_input = data.get("diary_entry", "").strip()
    if not user_input:
        return jsonify({"error": "Missing diary_entry"}), 400

    predicted_emotion = analyze_text_emotion(user_input)
    cluster_id = get_mood_cluster_id(predicted_emotion)
    songs_list = predict_song_scores(cluster_id, mood=predicted_emotion)

    return jsonify({
        "diary_entry": user_input,
        "predicted_emotion": predicted_emotion,
        "cluster_id": cluster_id,
        "songs": songs_list
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
