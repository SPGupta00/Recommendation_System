import os
import sys

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
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from transformers import pipeline  # <--- Added for Deep Learning NLP Extraction

# ─── Initialize R Environment ───────────────────────────────────────
r = robjects.r

# ─── File Path Strategy ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

MODELS_DIR = os.path.join(BASE_DIR, "r_models")
DATA_DIR = os.path.join(BASE_DIR, "data")
TEMPLATE_DIR = os.path.join(ROOT_DIR, "frontend", "templates")
STATIC_DIR = os.path.join(ROOT_DIR, "frontend", "static")

# ─── Load K-Means Bundle at Startup ──────────────────────────────────
r.assign("km_path", os.path.join(MODELS_DIR, "kmeans_songs.rds"))
r("km_bundle <- readRDS(km_path)")

with localconverter(robjects.default_converter + pandas2ri.converter):
    km_center = np.array(r("km_bundle$center"))
    km_scale = np.array(r("km_bundle$scale"))

# Primary Song Dataset
df_songs = pd.read_csv(os.path.join(DATA_DIR, "song_features.csv"))

# ─── Initialize AI Text Extractor Pipeline ─────────────────────────────
print("Loading Transformer Mood Extraction Engine...")
# This reads context, sentiment density, and nuances out-of-the-box
mood_extractor = pipeline(
    "text-classification", 
    model="j-hartmann/emotion-english-distilroberta-base", 
    top_k=1
)

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

def analyze_text_emotion(text: str) -> str:
    """Uses a Transformer network to dynamically extract the true core emotion keyword."""
    if not text.strip():
        return "neutral"
        
    try:
        # Run inference against the deep learning transformer pipeline
        predictions = mood_extractor(text)
        # Pull out the top predicted label keyword (e.g. 'joy', 'sadness', 'fear')
        extracted_keyword = predictions[0][0]['label']
        
        # Standardise model outputs to line up with your existing centroid dictionary
        mapping = {
            "joy": "happy",
            "sadness": "sad",
            "fear": "anxious",
            "surprise": "happy",
            "disgust": "angry",
            "neutral": "happy"
        }
        return mapping.get(extracted_keyword, extracted_keyword)
    except Exception as e:
        print(f"Extraction failed, dropping back to default baseline: {e}")
        return "happy"

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
    
    with localconverter(robjects.default_converter + pandas2ri.converter):
        centroids = np.array(r("km_bundle$model$centers"))
    
    distances = np.linalg.norm(centroids - scaled.reshape(1, -1), axis=1)
    return int(np.argmin(distances)) + 1  # 1-indexed matching R outputs

def predict_song_scores(cluster_id: int) -> list:
    """Ranks matching songs within the chosen cluster directly via database properties."""
    candidates = df_songs.copy()
    
    cluster_col = next((c for c in candidates.columns if c.lower() == 'cluster'), None)
    
    if cluster_col is None:
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_clusters = np.array(r("km_bundle$model$cluster"))
        
        if len(r_clusters) == len(candidates):
            candidates['cluster'] = r_clusters
            cluster_col = 'cluster'
        else:
            features = candidates[["valence", "energy", "danceability", "tempo", "acousticness"]].to_numpy()
            scaled_features = (features - km_center) / km_scale
            with localconverter(robjects.default_converter + pandas2ri.converter):
                centroids = np.array(r("km_bundle$model$centers"))
            
            assigned_list = []
            for row in scaled_features:
                dists = np.linalg.norm(centroids - row.reshape(1, -1), axis=1)
                assigned_list.append(int(np.argmin(dists)) + 1)
            candidates['cluster'] = assigned_list
            cluster_col = 'cluster'

    candidates = candidates[candidates[cluster_col] == cluster_id].copy()
    
    if candidates.empty:
        return []
        
    pop_col = next((c for c in candidates.columns if 'pop' in c.lower()), None)
    if pop_col:
        candidates["predicted_score"] = candidates[pop_col].astype(float)
    else:
        candidates["predicted_score"] = (candidates["valence"] * 50) + (candidates["energy"] * 50)

    t_key = next((k for k in candidates.columns if 'title' in k.lower() or 'name' in k.lower() or 'track' in k.lower()), None)
    a_key = next((k for k in candidates.columns if 'artist' in k.lower() or 'singer' in k.lower()), None)
    
    candidates = candidates.sort_values("predicted_score", ascending=False).reset_index(drop=True)
    
    ranked_songs = []
    for i, row in candidates.head(5).iterrows():
        song_title = row[t_key] if t_key else f"Track #{row.name}"
        song_artist = row[a_key] if a_key else "Unknown Artist"
        
        ranked_songs.append({
            "rank": i + 1,
            "title": song_title,
            "artist": song_artist,
            "tempo": int(row.get("tempo", 0)) if "tempo" in row else 120,
            "energy": row.get("energy", 0) if "energy" in row else 0.5,
            "predicted_score": row["predicted_score"]
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
            songs_list = predict_song_scores(cluster_id)
            
    return render_template(
        "index.html",
        songs=songs_list,
        user_input=user_input,
        predicted_emotion=predicted_emotion.capitalize(),
        cluster_id=cluster_id
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
