# ─── Supervised Model: Random Forest (Songs Edition) ─────────
# Predicts popularity_score based on your K-Means cluster and features.
# Run this once in R to rewrite: backend/r_models/mood_rf.rds

library(randomForest)

# Load song features dataset
songs <- read.csv("backend/data/song_features.csv", stringsAsFactors = FALSE)

# Load the saved K-Means model to apply cluster tags dynamically
km_bundle <- readRDS("backend/r_models/kmeans_songs.rds")
songs$cluster <- km_bundle$model$cluster

# Convert structural cluster numbers to categories (factors)
songs$cluster <- as.factor(songs$cluster)

# Split into 80% Training and 20% Testing groups
set.seed(42)
n <- nrow(songs)
train_idx <- sample(n, floor(n * 0.8))
train_df  <- songs[train_idx, ]
test_df   <- songs[-train_idx, ]

# Train the fresh model using song attributes
rf_model <- randomForest(
  popularity_score ~ cluster + valence + energy + danceability + tempo + acousticness,
  data     = train_df,
  ntree    = 200,
  importance = TRUE
)

# Verify performance criteria
preds <- predict(rf_model, newdata = test_df)
rmse  <- sqrt(mean((preds - test_df$popularity_score)^2))
cat(sprintf("Success! Model Test RMSE: %.2f\n", rmse))

# Save over the old file with the new song architecture
dir.create("backend/r_models", recursive = TRUE, showWarnings = FALSE)
saveRDS(rf_model, "backend/r_models/mood_rf.rds")
cat("New Song Random Forest Model saved successfully!\n")
