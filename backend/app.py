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

# ─── Initialize R Environment ───────────────────────────────────────
r = robjects.r
r.library("randomForest")
try:
    r.library("stats")
except Exception:
    pass 

# ─── Paths ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# Backend sub-folders
MODELS_DIR = os.path.join(BASE_DIR, "r_models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Frontend sibling folders
TEMPLATE_DIR = os.path.join(ROOT_DIR, "frontend", "templates")
STATIC_DIR = os.path.join(ROOT_DIR, "frontend", "static")

# ─── Load Models & Data at Startup ──────────────────────────────────
# Load the Random Forest model normally
rf_model = r.readRDS(os.path.join(MODELS_DIR, "mood_rf.rds"))

# Assign the file path to R and let R read components directly
r.assign("km_path", os.path.join(MODELS_DIR, "kmeans_songs.rds"))
r("km_bundle <- readRDS(km_path)")

# Wrap initial vector extractions in localconverter to ensure rule safety
with localconverter(robjects.default_converter + pandas2ri.converter):
    km_center = np.array(r("km_bundle$center"))
    km_scale = np.array(r("km_bundle$scale"))

# Load activity data
df = pd.read_csv(os.path.join(DATA_DIR, "mood_activities.csv"))
moods = sorted(df["mood"].unique().tolist())

# ─── Flask App Setup ────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

def predict_activity_scores(mood: str) -> pd.DataFrame:
    """Use R Random Forest to predict + rank activities for a mood."""
    candidates = df[df["mood"] == mood].copy()
    
    # FIXED: Re-enforcing the localconverter on the active thread context
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_df = pandas2ri.py2rpy(candidates[["mood", "intensity", "duration_min"]])
        
        mood_levels = robjects.StrVector(sorted(df["mood"].unique()))
        intensity_levels = robjects.StrVector(sorted(df["intensity"].unique()))
        
        r.assign("temp_df", r_df)
        r.assign("m_levs", mood_levels)
        r.assign("i_levs", intensity_levels)
        r("temp_df$mood <- factor(temp_df$mood, levels = m_levs)")
        r("temp_df$intensity <- factor(temp_df$intensity, levels = i_levs)")
        
        r.assign("model", rf_model)
        r("preds <- predict(model, temp_df)")
        scores = np.array(r["preds"])

    candidates["predicted_score"] = scores
    candidates = candidates.sort_values("predicted_score", ascending=False).reset_index(drop=True)
    candidates["rank"] = range(1, len(candidates) + 1)
    return candidates.head(5)

def get_song_cluster(mood: str) -> dict:
    """Assign mood feature vector to nearest K-Means centroid."""
    mood_centroids = {
        "happy": [0.8, 0.7, 0.8, 130, 0.1],
        "sad": [0.2, 0.2, 0.4, 70, 0.8],
        "stressed": [0.6, 0.8, 0.7, 140, 0.1],
        "anxious": [0.4, 0.3, 0.5, 80, 0.6],
        "angry": [0.7, 0.9, 0.8, 145, 0.05],
        "tired": [0.4, 0.2, 0.5, 75, 0.7],
        "bored": [0.6, 0.6, 0.7, 120, 0.2],
    }
    
    if mood not in mood_centroids:
        return {}
        
    raw = np.array(mood_centroids[mood])
    scaled = (raw - km_center) / km_scale
    
    # FIXED: Wrapped in localconverter to safely process vectors on request threads
    with localconverter(robjects.default_converter + pandas2ri.converter):
        centroids = np.array(r("km_bundle$model$centers"))
    
    distances = np.linalg.norm(centroids - scaled.reshape(1, -1), axis=1)
    cluster_id = int(np.argmin(distances)) + 1  
    return {"cluster": cluster_id, "mood": mood}

# ─── App Routes ─────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    activities = None
    user_mood = ""
    song_cluster = None
    
    if request.method == "POST":
        user_mood = request.form.get("mood", "").strip().lower()
        if user_mood and user_mood in moods:
            activities = predict_activity_scores(user_mood).to_dict("records")
            song_cluster = get_song_cluster(user_mood)
            
    return render_template(
        "index.html",
        moods=moods,
        activities=activities,
        user_mood=user_mood,
        song_cluster=song_cluster
    )

@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """JSON API endpoint."""
    data = request.get_json()
    mood = data.get("mood", "").strip().lower()
    
    if mood not in moods:
        return jsonify({"error": "Invalid mood"}), 400
        
    activities = predict_activity_scores(mood)
    song_cluster = get_song_cluster(mood)
    
    return jsonify({
        "mood": mood,
        "activities": activities[["rank", "activity", "intensity", "duration_min", "predicted_score"]].to_dict("records"),
        "song_cluster": song_cluster
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
