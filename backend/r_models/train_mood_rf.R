# =====================================================================
# ─── PART 2: Supervised Model (Random Forest) ────────────────────────
# =====================================================================
# Trains on popularity to rank songs inside the filtered clusters
# Run once to generate: backend/r_models/mood_rf.rds

library(randomForest)

# ─── LOAD NEW DATASET & CLUSTER BUNDLE ──────────────────────────────
songs <- read.csv("backend/data/song_data.csv", stringsAsFactors = FALSE)
if (!file.exists("backend/r_models/kmeans_songs.rds")) {
  stop("Error: kmeans_songs.rds missing! Please run the K-Means script first.")
}
km_bundle <- readRDS("backend/r_models/kmeans_songs.rds")

# Normalize column names to map text values cleanly
colnames(songs) <- tolower(trimws(colnames(songs)))
for (i in seq_along(colnames(songs))) {
  if (colnames(songs)[i] %in% c("valence", "audio_valence", "audio_valencies")) colnames(songs)[i] <- "audio_valence"
  if (colnames(songs)[i] %in% c("energy", "audio_energy")) colnames(songs)[i] <- "energy"
  if (colnames(songs)[i] %in% c("danceability", "danceable")) colnames(songs)[i] <- "danceability"
  if (colnames(songs)[i] %in% c("tempo", "bpm", "speed")) colnames(songs)[i] <- "tempo"
  if (colnames(songs)[i] %in% c("acousticness", "acoustic")) colnames(songs)[i] <- "acousticness"
  if (colnames(songs)[i] %in% c("song_popularity", "popularity", "track_popularity")) colnames(songs)[i] <- "song_popularity"
  if (colnames(songs)[i] %in% c("duration_min", "duration", "length", "duration_ms")) colnames(songs)[i] <- "duration_min"
}

# Inject cluster tracking fields dynamically from the clustering matrix output
songs$cluster <- km_bundle$model$cluster
songs$cluster <- as.factor(songs$cluster)

# Fallback metric safety rules for unmapped metrics
if (!"song_popularity" %in% colnames(songs)) {
  cat("Warning: 'song_popularity' column missing. Generating dummy popularity scores.\n")
  songs$song_popularity <- runif(nrow(songs), min = 10, max = 100)
}
if (!"duration_min" %in% colnames(songs)) {
  cat("Warning: 'duration_min' column missing. Defaulting values to 3.5.\n")
  songs$duration_min <- 3.5
} else if (max(songs$duration_min, na.rm = TRUE) > 1000) {
  cat("Notice: Converting duration column values from milliseconds to minutes.\n")
  songs$duration_min <- songs$duration_min / 60000.0
}

# Split into 80% Training and 20% Testing groups
set.seed(42)
n <- nrow(songs)
train_idx <- sample(n, floor(n * 0.8))
train_df  <- songs[train_idx, ]
test_df   <- songs[-train_idx, ]

# ─── TRAIN RANDOM FOREST MODEL ───────────────────────────────────────
rf_model <- randomForest(
  song_popularity ~ cluster + audio_valence + energy + danceability + tempo + acousticness + duration_min,
  data       = train_df,
  ntree      = 200,
  importance = TRUE
)

# Verify testing performance criteria
preds <- predict(rf_model, newdata = test_df)
rmse  <- sqrt(mean((preds - test_df$song_popularity)^2, na.rm = TRUE))
cat(sprintf("\nSuccess! Random Forest Model Test RMSE: %.2f\n", rmse))

# Save over the old model layout with the fully functional track architecture
saveRDS(rf_model, "backend/r_models/mood_rf.rds")
cat("New Song Random Forest Model saved successfully to backend/r_models/mood_rf.rds\n")
