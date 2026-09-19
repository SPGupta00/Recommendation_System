# ─── Supervised Model: Random Forest ─────────────────────────
# Predicts popularity_score for a given (mood, activity) pair
# Run once to generate: backend/r_models/mood_rf.rds

library(randomForest)
library(dplyr)

# Load data
df <- read.csv("Recommendation_System/backend/data/mood_activities.csv", stringsAsFactors = FALSE)

# Convert to factors
df$mood      <- as.factor(df$mood)
df$intensity <- as.factor(df$intensity)

# Split 80/20
set.seed(42)
n <- nrow(df)
train_idx <- sample(n, floor(n * 0.8))
train_df  <- df[train_idx, ]
test_df   <- df[-train_idx, ]

# Train
rf_model <- randomForest(
  popularity_score ~ mood + intensity + duration_min,
  data     = train_df,
  ntree    = 200,
  importance = TRUE
)

# Evaluate
preds <- predict(rf_model, newdata = test_df)
rmse  <- sqrt(mean((preds - test_df$popularity_score)^2))
cat(sprintf("Test RMSE: %.2f\n", rmse))
cat("Variable Importance:\n")
print(importance(rf_model))

# Save
dir.create("Recommendation_System/backend/r_models", recursive = TRUE, showWarnings = FALSE)
saveRDS(rf_model, "Recommendation_System/backend/r_models/mood_rf.rds")
cat("Model saved to Recommendation_System/backend/r_models/mood_rf.rds\n")   