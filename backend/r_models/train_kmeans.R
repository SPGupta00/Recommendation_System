# =====================================================================
# ─── PART 1: Unsupervised Model (K-Means Clustering) ─────────────────
# =====================================================================
# Clusters songs by audio features into mood groups
# Run once to generate: backend/r_models/kmeans_songs.rds

library(cluster)

# Ensure output directory structure exists
dir.create("backend/r_models", recursive = TRUE, showWarnings = FALSE)

# ─── LOAD NEW DATASET ────────────────────────────────────────────────
if (!file.exists("backend/data/song_data.csv")) {
  stop("Error: backend/data/song_data.csv not found! Check your file path.")
}
songs <- read.csv("backend/data/song_data.csv", stringsAsFactors = FALSE)

# Normalize column names to handle naming variations automatically
colnames(songs) <- tolower(trimws(colnames(songs)))
for (i in seq_along(colnames(songs))) {
  if (colnames(songs)[i] %in% c("valence", "audio_valence", "audio_valencies")) colnames(songs)[i] <- "audio_valence"
  if (colnames(songs)[i] %in% c("energy", "audio_energy")) colnames(songs)[i] <- "energy"
  if (colnames(songs)[i] %in% c("danceability", "danceable")) colnames(songs)[i] <- "danceability"
  if (colnames(songs)[i] %in% c("tempo", "bpm", "speed")) colnames(songs)[i] <- "tempo"
  if (colnames(songs)[i] %in% c("acousticness", "acoustic")) colnames(songs)[i] <- "acousticness"
}

# Verify columns exist before clustering
feature_cols <- c("audio_valence", "energy", "danceability", "tempo", "acousticness")
missing_cols <- setdiff(feature_cols, colnames(songs))
if (length(missing_cols) > 0) {
  stop(paste("Error: Missing required audio feature columns in CSV:", paste(missing_cols, collapse = ", ")))
}

# Extract standardized feature matrix
features <- songs[, feature_cols]

# Scale features
features_scaled <- scale(features)

# Save scaling parameters (CRITICAL for web pipeline prediction consistency)
scaling_center <- attr(features_scaled, "scaled:center")
scaling_scale  <- attr(features_scaled, "scaled:scale")

# Train final clustering model (k=5)
set.seed(42)
km_model <- kmeans(features_scaled, centers = 5, nstart = 25, iter.max = 100)

# Evaluate sizes
cat("\n--- K-Means Training Completed ---\nCluster sizes:\n")
print(table(km_model$cluster))

# Save cluster package bundle together 
saveRDS(list(
  model  = km_model,
  center = scaling_center,
  scale  = scaling_scale
), "backend/r_models/kmeans_songs.rds")

cat("K-Means Bundle successfully saved to backend/r_models/kmeans_songs.rds\n")
