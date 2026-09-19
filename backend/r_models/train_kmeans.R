# ─── Unsupervised Model: K-Means Clustering ──────────────────
# Clusters songs by audio features into mood groups
# Run once to generate: backend/models/kmeans_songs.rds

library(cluster)

# Load song features
songs <- read.csv("backend/data/song_features.csv", stringsAsFactors = FALSE)

# Feature matrix
features <- songs[, c("valence", "energy", "danceability", "tempo", "acousticness")]

# Scale features
features_scaled <- scale(features)

# ── Save scaling params (CRITICAL for prediction) ────────────
scaling_center <- attr(features_scaled, "scaled:center")
scaling_scale  <- attr(features_scaled, "scaled:scale")

# Elbow method to find optimal k
wss <- sapply(2:8, function(k) {
  km <- kmeans(features_scaled, centers = k, nstart = 25, iter.max = 100)
  km$tot.withinss
})

plot(2:8, wss, type = "b", xlab = "K", ylab = "Within-Cluster SS",
     main = "Elbow Method")

# Train final model (k=5)
set.seed(42)
km_model <- kmeans(features_scaled, centers = 5, nstart = 25, iter.max = 100)

# Add cluster labels to song data
songs$cluster <- km_model$cluster

# Evaluate
cat("Cluster sizes:\n")
print(table(km_model$cluster))

sil <- silhouette(km_model$cluster, dist(features_scaled))
cat(sprintf("Average Silhouette: %.3f\n", mean(sil[, 3])))

# ── Save model + scaling params together ─────────────────────
saveRDS(list(
  model  = km_model,
  center = scaling_center,
  scale  = scaling_scale
), "backend/r_models/kmeans_songs.rds")

cat("Model saved to backend/r_models/kmeans_songs.rds\n")   