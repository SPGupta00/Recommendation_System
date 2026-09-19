# ─── Unsupervised Model: K-Means Clustering ────────────────── #
# Clusters songs by audio features into mood groups
# Run once to generate: backend/r_models/kmeans_songs.rds

# Install required packages if you don't have them yet:
# install.packages(c("cluster", "here"))

library(cluster)
library(here) # Resolves working directory path mismatch errors automatically

cat("Project Root identified as:", here(), "\n")
cat("Loading song features dataset...\n")

# Load song features safely using absolute paths generated relative to project root
songs_path <- here("Recommendation_System","backend", "data", "song_features.csv")
if (!file.exists(songs_path)) {
  stop("ERROR: The file 'song_features.csv' was not found at: ", songs_path, 
       "\nPlease check your folder spelling or layout!")
}
songs <- read.csv(songs_path, stringsAsFactors = FALSE)

# Feature matrix extraction
features <- songs[, c("valence", "energy", "danceability", "tempo", "acousticness")]

# Scale features to avoid distance bias
features_scaled <- scale(features)

# ── Save scaling params (CRITICAL for real-time app inference) ────────────
scaling_center <- attr(features_scaled, "scaled:center")
scaling_scale <- attr(features_scaled, "scaled:scale")

# Elbow method to verify optimal cluster sizing (k)
cat("Calculating Within-Cluster Sum of Squares (Elbow method)...\n")
wss <- sapply(2:8, function(k) {
  km <- kmeans(features_scaled, centers = k, nstart = 25, iter.max = 100)
  km$tot.withinss
})

# Save the elbow plot to a dedicated file inside your project structure
pdf(here("Rplots.pdf"))
plot(2:8, wss, type = "b", xlab = "K", ylab = "Within-Cluster SS", main = "Elbow Method Evaluation")
dev.off()
cat("Elbow method plot generated and updated in your project root as 'Rplots.pdf'\n")

# Train final K-Means model (k=5 mood groups)
cat("Training final K-Means cluster engine (k=5)...\n")
set.seed(42)
km_model <- kmeans(features_scaled, centers = 5, nstart = 25, iter.max = 100)

# Inject the generated cluster classifications back into your dataframe memory matrix
songs$cluster <- km_model$cluster

# Write the clustered songs backend data over your existing CSV so Python can access the tags
write.csv(songs, songs_path, row.names = FALSE)
cat("Updated 'song_features.csv' with cluster identification flags.\n")

# Evaluate separation quality via Silhouettes
cat("\n--- Evaluation Metrics ---\n")
cat("Cluster sizes:\n")
print(table(km_model$cluster))

sil <- silhouette(km_model$cluster, dist(features_scaled))
cat(sprintf("Average Separation Silhouette Score: %.3f\n", mean(sil[, 3])))

# ── Save model parameters to your model folder directory targets ────────────
output_model_path <- here("Recommendation_System","backend", "r_models", "kmeans_songs.rds")
saveRDS(list(
  model = km_model,
  center = scaling_center,
  scale = scaling_scale
), output_model_path)

cat("\n[SUCCESS] Pipeline Complete. Model bundle saved to:", output_model_path, "\n")
