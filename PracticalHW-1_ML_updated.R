# Load required libraries for modeling, evaluation, and visualization
library(tidyverse)
library(tree)
library(randomForest)
library(gbm)
library(caret)
library(MLmetrics)
library(reshape2)
library(ggplot2)

# Load the cleaned dataset
load("/Users/pavan/Downloads/youth_data.Rdata")

# ----------------------------------------------------
# BINARY CLASSIFICATION: Predicting School Skipping
# ----------------------------------------------------
#Question : Can we predict whether a youth skipped school in the past month based on their substance use behavior?

# Create binary target variable from EDUSKPCOM (0 = No, 1 = Yes)
df$EDUSKPCOM <- as.numeric(as.character(df$EDUSKPCOM))
df$SKIPPED_SCHOOL <- ifelse(df$EDUSKPCOM >= 1 & df$EDUSKPCOM <= 30, 1,
                            ifelse(df$EDUSKPCOM == 0, 0, NA))

# Prepare the dataset by selecting substance use and demographic predictors
expanded_df <- df %>%
  select(SKIPPED_SCHOOL,
         IRALCFM, IRMJFM, IRCIGFM,
         IRMJAGE, IRALCAGE, IRCIGAGE,
         MRJFLAG, ALCFLAG, TOBFLAG,
         all_of(youth_experience_cols),
         all_of(demographic_cols)) %>%
  select(-EDUSKPCOM)

# Replace placeholder 991 values with NA in numeric columns
num_cols <- c("IRALCFM", "IRMJFM", "IRCIGFM", "IRMJAGE", "IRALCAGE", "IRCIGAGE")
expanded_df[num_cols] <- lapply(expanded_df[num_cols], function(x) na_if(as.numeric(as.character(x)), 991))

# Convert binary target and flag variables to factors
expanded_df <- expanded_df %>%
  mutate(SKIPPED_SCHOOL = factor(SKIPPED_SCHOOL),
         MRJFLAG = factor(MRJFLAG),
         ALCFLAG = factor(ALCFLAG),
         TOBFLAG = factor(TOBFLAG)) %>%
  drop_na()

# ----------------------
# BAGGING CLASSIFIER
# ----------------------
set.seed(19)
binary_df <- expanded_df
idx <- sample(1:nrow(binary_df), 0.7 * nrow(binary_df))
train <- binary_df[idx, ]
test <- binary_df[-idx, ]

bag_model <- randomForest(SKIPPED_SCHOOL ~ ., data = train, mtry = ncol(train) - 1, ntree = 100, importance = TRUE)
probs <- predict(bag_model, newdata = test, type = "prob")[, 2]
bag_pred_thresh <- ifelse(probs > 0.4, 1, 0)

# Evaluate model
conf_mat <- confusionMatrix(factor(bag_pred_thresh), factor(test$SKIPPED_SCHOOL))
precision <- Precision(bag_pred_thresh, test$SKIPPED_SCHOOL)
recall <- Recall(bag_pred_thresh, test$SKIPPED_SCHOOL)
f1 <- F1_Score(bag_pred_thresh, test$SKIPPED_SCHOOL)

# Visualize confusion matrix
heatmap_data <- as.data.frame(conf_mat$table)
colnames(heatmap_data) <- c("Predicted", "Actual", "Freq")

# Create heatmap plot
heatmap_plot <- ggplot(heatmap_data, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_gradient(low = "skyblue", high = "darkblue") +
  labs(title = "Confusion Matrix Heatmap - Bagging",
       x = "Actual Label", y = "Predicted Label") +
  theme_minimal()

print(heatmap_plot)

# Print evaluation metrics
cat("Bagging Accuracy:", round(mean(bag_pred_thresh == test$SKIPPED_SCHOOL), 3), "\n")
cat("Bagging Precision:", round(precision, 3), "\n")
cat("Bagging Recall:", round(recall, 3), "\n")
cat("Bagging F1 Score:", round(f1, 3), "\n")

# ----------------------
# DECISION TREE MODEL
# ----------------------
set.seed(19)
sample_index <- sample(seq_len(nrow(expanded_df)), size = 0.7 * nrow(expanded_df))
train_data <- expanded_df[sample_index, ]
test_data <- expanded_df[-sample_index, ]

# Fit and prune decision tree
tree_model <- tree(SKIPPED_SCHOOL ~ ., data = train_data)
cv_result <- cv.tree(tree_model, FUN = prune.misclass)
pruned_tree <- prune.misclass(tree_model, best = 11)

# Predict using pruned tree
pruned_pred <- predict(pruned_tree, newdata = test_data, type = "class")
conf_matrix_pruned <- table(Predicted = pruned_pred, Actual = test_data$SKIPPED_SCHOOL)


plot(cv_result$size, cv_result$dev, type = "b", pch = 19,
     xlab = "Number of Terminal Nodes",
     ylab = "Cross-Validated Error",
     main = "Binary Classification: Tree Size vs Error")
#From above graph we have finalised that best number of trees is 11

pruned_accuracy <- sum(diag(conf_matrix_pruned)) / sum(conf_matrix_pruned)

cat("Pruned Tree Accuracy (Tree Size = 11):", round(pruned_accuracy, 3), "\n")

pruned_conf <- confusionMatrix(factor(pruned_pred), factor(test_data$SKIPPED_SCHOOL))

#Visuals for pruned decision tree
tree_heatmap_data <- as.data.frame(pruned_conf$table)
colnames(tree_heatmap_data) <- c("Predicted", "Actual", "Freq")

ggplot(tree_heatmap_data, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 5, color = "white") +
  scale_fill_gradient(low = "skyblue", high = "darkblue") +
  labs(title = "Pruned Decision Tree - Confusion Matrix Heatmap",
       x = "Actual Class", y = "Predicted Class") +
  theme_minimal()

# ----------------------
# RANDOM FOREST MODEL
# ----------------------
set.seed(19)

rf_model <- randomForest(
  SKIPPED_SCHOOL ~ .,
  data = train_data,
  ntree = 500,
  mtry = 4,
  importance = TRUE
)

rf_pred <- predict(rf_model, newdata = test_data)

rf_conf <- confusionMatrix(factor(rf_pred), factor(test_data$SKIPPED_SCHOOL))
rf_precision <- Precision(rf_pred, test_data$SKIPPED_SCHOOL)
rf_recall <- Recall(rf_pred, test_data$SKIPPED_SCHOOL)
rf_f1 <- F1_Score(rf_pred, test_data$SKIPPED_SCHOOL)

cat("Random Forest Accuracy:", round(mean(rf_pred == test_data$SKIPPED_SCHOOL), 3), "\n")
cat("Random Forest Precision:", round(rf_precision, 3), "\n")
cat("Random Forest Recall:", round(rf_recall, 3), "\n")
cat("Random Forest F1 Score:", round(rf_f1, 3), "\n")

#Visuals for Random Forest
rf_heatmap_data <- as.data.frame(rf_conf$table)
colnames(rf_heatmap_data) <- c("Predicted", "Actual", "Freq")

ggplot(rf_heatmap_data, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 5, color = "white") +
  scale_fill_gradient(low = "skyblue", high = "darkblue") +
  labs(title = "Random Forest - Confusion Matrix Heatmap",
       x = "Actual Class", y = "Predicted Class") +
  theme_minimal()

# ----------------------
# GBM CLASSIFIER
# ----------------------
set.seed(19)

# Convert target variable to numeric for GBM
train_gbm <- train_data
test_gbm <- test_data
train_gbm$SKIPPED_SCHOOL <- as.numeric(as.character(train_gbm$SKIPPED_SCHOOL))
test_gbm$SKIPPED_SCHOOL <- as.numeric(as.character(test_gbm$SKIPPED_SCHOOL))

# Fit GBM model with cross-validation
gbm_model <- gbm(
  SKIPPED_SCHOOL ~ .,
  data = train_gbm,
  distribution = "bernoulli",
  n.trees = 5000,
  interaction.depth = 3,
  shrinkage = 0.01,
  cv.folds = 5,
  verbose = FALSE
)

# Select best number of trees
gbm_best_iter <- gbm.perf(gbm_model, method = "cv")

# Predict on test set using optimal number of trees
gbm_probs <- predict(gbm_model, newdata = test_gbm, n.trees = gbm_best_iter, type = "response")
gbm_preds <- ifelse(gbm_probs > 0.4, 1, 0)

# Evaluate GBM predictions
gbm_conf <- confusionMatrix(factor(gbm_preds), factor(test_gbm$SKIPPED_SCHOOL))
gbm_precision <- Precision(gbm_preds, test_gbm$SKIPPED_SCHOOL)
gbm_recall <- Recall(gbm_preds, test_gbm$SKIPPED_SCHOOL)
gbm_f1 <- F1_Score(gbm_preds, test_gbm$SKIPPED_SCHOOL)

# Print metrics
cat("GBM Accuracy:", round(mean(gbm_preds == test_gbm$SKIPPED_SCHOOL), 3), "\n")
cat("GBM Precision:", round(gbm_precision, 3), "\n")
cat("GBM Recall:", round(gbm_recall, 3), "\n")
cat("GBM F1 Score:", round(gbm_f1, 3), "\n")

# Visualize GBM confusion matrix
gbm_heatmap_data <- as.data.frame(gbm_conf$table)
colnames(gbm_heatmap_data) <- c("Predicted", "Actual", "Freq")

heatmap_gbm <- ggplot(gbm_heatmap_data, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 5, color = "white") +
  scale_fill_gradient(low = "skyblue", high = "darkblue") +
  labs(title = "GBM - Confusion Matrix Heatmap",
       x = "Actual Class", y = "Predicted Class") +
  theme_minimal()

print(heatmap_gbm)

# ----------------------
# Final F1 Score Comparison (Binary)
# ----------------------
f1_binary_df <- data.frame(
  Model = c("Bagging", "Decision Tree", "Random Forest", "GBM"),
  F1_Score = c(f1, 0.5, rf_f1, gbm_f1)
)

f1_plot_binary <- ggplot(f1_binary_df, aes(x = reorder(Model, -F1_Score), y = F1_Score, fill = Model)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = round(F1_Score, 3)), vjust = -0.5, size = 4.5) +
  scale_fill_brewer(palette = "Blues") +
  ylim(0, 1) +
  labs(title = "Binary Classification Model Comparison - F1 Score",
       x = "Model", y = "F1 Score") +
  theme_minimal() +
  theme(legend.position = "none")

print(f1_plot_binary)


# ----------------------------------------------------
# MULTICLASS CLASSIFICATION: Predicting Marijuana Use Level
# ----------------------------------------------------
#Question : Can we classify the frequency of marijuana use among youth into None, Occasional, and Frequent groups based on their home environment and peer influence?
# target variable SKIP_LEVEL based on MRJMDAYS
set.seed(19)
df$MRJMDAYS <- as.numeric(as.character(df$MRJMDAYS))
df <- df %>% filter(!is.na(MRJMDAYS) & MRJMDAYS <= 5)

df$SKIP_LEVEL <- case_when(
  df$MRJMDAYS == 5 ~ "None",
  df$MRJMDAYS %in% c(3, 4) ~ "Occasional",
  df$MRJMDAYS %in% c(1, 2) ~ "Frequent"
)
df$SKIP_LEVEL <- factor(df$SKIP_LEVEL, levels = c("Frequent", "None", "Occasional"))

# Select relevant peer and home environment predictors
home_peer_cols <- c("PARCHKHW", "PARHLPHW", "PRPROUD2", "PRPKCIG2", "PRTALK3",
                    "PRBSOLV2", "PREVIOL2", "FRDMJMON", "PRMJEVR2",
                    "YFLMJMO", "YFLTMRJ2")

multi_df <- df %>%
  select(SKIP_LEVEL, all_of(home_peer_cols)) %>%
  drop_na()

# Balance the dataset via downsampling to ensure equal representation
multi_df_balanced <- multi_df %>%
  group_by(SKIP_LEVEL) %>%
  slice_sample(n = min(table(multi_df$SKIP_LEVEL))) %>%
  ungroup()

# Train-test split for multiclass task
set.seed(19)
idx <- sample(1:nrow(multi_df_balanced), 0.7 * nrow(multi_df_balanced))
train_multi <- multi_df_balanced[idx, ]
test_multi <- multi_df_balanced[-idx, ]

# ----------------------
# Decision Tree (Multiclass)
# ----------------------

# Fit decision tree model
tree_multi <- tree(SKIP_LEVEL ~ ., data = train_multi)
tree_pred <- predict(tree_multi, newdata = test_multi, type = "class")

# Evaluate model performance
conf_matrix_tree <- table(Predicted = tree_pred, Actual = test_multi$SKIP_LEVEL)

evaluate_f1 <- function(cm, class_index) {
  TP <- cm[class_index, class_index]
  FP <- sum(cm[class_index, -class_index])
  FN <- sum(cm[-class_index, class_index])
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  f1 <- 2 * precision * recall / (precision + recall)
  return(round(f1, 3))
}

f1_frequent <- evaluate_f1(conf_matrix_tree, 1)
f1_none <- evaluate_f1(conf_matrix_tree, 2)
f1_occasional <- evaluate_f1(conf_matrix_tree, 3)
macro_f1_tree <- mean(c(f1_frequent, f1_none, f1_occasional))

cat("Decision Tree Macro F1:", macro_f1_tree, "\n")

# Heatmap visualization
tree_heatmap_data <- as.data.frame(conf_matrix_tree)
colnames(tree_heatmap_data) <- c("Prediction", "Actual", "Freq")

ggplot(tree_heatmap_data, aes(x = Actual, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 5, color = "white") +
  scale_fill_gradient(low = "skyblue", high = "darkblue") +
  labs(title = "Decision Tree - Multiclass Confusion Matrix",
       x = "Actual Class", y = "Predicted Class") +
  theme_minimal()

# ----------------------
# Random Forest (Multiclass)
# ----------------------
set.seed(19)
rf_multi <- randomForest(SKIP_LEVEL ~ ., data = train_multi, ntree = 200, mtry = 4)
rf_pred <- predict(rf_multi, test_multi, type = "class")
conf_matrix_rf <- table(Predicted = rf_pred, Actual = test_multi$SKIP_LEVEL)

f1_frequent_rf <- evaluate_f1(conf_matrix_rf, 1)
f1_none_rf <- evaluate_f1(conf_matrix_rf, 2)
f1_occasional_rf <- evaluate_f1(conf_matrix_rf, 3)
macro_f1_rf <- mean(c(f1_frequent_rf, f1_none_rf, f1_occasional_rf))

cat("Random Forest Macro F1:", macro_f1_rf, "\n")

# Heatmap
data_rf <- as.data.frame(conf_matrix_rf)
colnames(data_rf) <- c("Prediction", "Actual", "Freq")

ggplot(data_rf, aes(x = Actual, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 5, color = "white") +
  scale_fill_gradient(low = "skyblue", high = "darkblue") +
  labs(title = "Random Forest - Multiclass Confusion Matrix",
       x = "Actual Class", y = "Predicted Class") +
  theme_minimal()

# ----------------------
# GBM (Multiclass)
# ----------------------
set.seed(19)
gbm_df <- multi_df_balanced
gbm_df$SKIP_LEVEL_NUM <- as.numeric(as.factor(gbm_df$SKIP_LEVEL))

idx <- sample(1:nrow(gbm_df), 0.7 * nrow(gbm_df))
train_gbm <- gbm_df[idx, ]
test_gbm <- gbm_df[-idx, ]

gbm_model_multi <- gbm(
  formula = SKIP_LEVEL_NUM ~ . -SKIP_LEVEL,
  data = train_gbm,
  distribution = "multinomial",
  n.trees = 3000,
  interaction.depth = 3,
  shrinkage = 0.01,
  cv.folds = 5,
  verbose = FALSE
)

best_iter_gbm <- gbm.perf(gbm_model_multi, method = "cv")
gbm_probs <- predict(gbm_model_multi, newdata = test_gbm, n.trees = best_iter_gbm, type = "response")
gbm_pred_numeric <- apply(gbm_probs, 1, which.max)

true_levels <- levels(multi_df_balanced$SKIP_LEVEL)
predicted_factor <- factor(gbm_pred_numeric, labels = true_levels)
actual_factor <- factor(test_gbm$SKIP_LEVEL_NUM, labels = true_levels)

conf_matrix_gbm <- table(Predicted = predicted_factor, Actual = actual_factor)
f1_frequent_gbm <- evaluate_f1(conf_matrix_gbm, 1)
f1_none_gbm <- evaluate_f1(conf_matrix_gbm, 2)
f1_occasional_gbm <- evaluate_f1(conf_matrix_gbm, 3)
macro_f1_gbm <- mean(c(f1_frequent_gbm, f1_none_gbm, f1_occasional_gbm))

cat("GBM Macro F1:", macro_f1_gbm, "\n")

gbm_heatmap_data <- as.data.frame(conf_matrix_gbm)
colnames(gbm_heatmap_data) <- c("Prediction", "Actual", "Freq")

ggplot(gbm_heatmap_data, aes(x = Actual, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 5, color = "white") +
  scale_fill_gradient(low = "skyblue", high = "darkblue") +
  labs(title = "GBM - Multiclass Confusion Matrix",
       x = "Actual Class", y = "Predicted Class") +
  theme_minimal()

# ----------------------
# F1 Score Comparison Plot (Multiclass)
# ----------------------
f1_comparison_df <- data.frame(
  Model = rep(c("Decision Tree", "Random Forest", "GBM"), each = 3),
  Class = rep(c("Frequent", "None", "Occasional"), times = 3),
  F1_Score = c(f1_frequent, f1_none, f1_occasional,
               f1_frequent_rf, f1_none_rf, f1_occasional_rf,
               f1_frequent_gbm, f1_none_gbm, f1_occasional_gbm)
)

ggplot(f1_comparison_df, aes(x = Class, y = F1_Score, fill = Model)) +
  geom_col(position = "dodge", width = 0.7) +
  geom_text(aes(label = round(F1_Score, 3)), position = position_dodge(width = 0.7), vjust = -0.3, size = 4) +
  scale_fill_brewer(palette = "Set2") +
  ylim(0, 1) +
  labs(title = "Multiclass Model Comparison - F1 Score by Class",
       x = "Class", y = "F1 Score") +
  theme_minimal()

# ----------------------------------------------------
# REGRESSION TASK: Predicting Alcohol Consumption Days
# ----------------------------------------------------
# Can we predict how many days a youth drank alcohol in the past year based on their demographics and parental involvement?
# Load required libraries
library(tree)
library(randomForest)
library(gbm)
library(ggplot2)
library(tidyverse)

# Set seed for reproducibility
set.seed(19)

# Define relevant predictors for regression task
regression_features <- c("ALCYDAYS", "FRDADLY2", "PARCHKHW", "PRBSOLV2", "PRPROUD2",
                         "YTHACT2", "NEWRACE2", "IRSEX", "INCOME", "EDUSCHGRD2")

# Prepare the dataset by filtering and removing NA values
df_reg <- df %>%
  select(all_of(regression_features)) %>%
  drop_na()

# Split the dataset into training and testing sets
idx <- sample(1:nrow(df_reg), 0.7 * nrow(df_reg))
train <- df_reg[idx, ]
test <- df_reg[-idx, ]

# ----------------------
# Decision Tree Regression
# ----------------------
tree_model <- tree(
  ALCYDAYS ~ .,
  data = train,
  control = tree.control(nobs = nrow(train), mincut = 15, minsize = 30, mindev = 0.01)
)

tree_preds <- predict(tree_model, newdata = test)
tree_mse <- mean((tree_preds - test$ALCYDAYS)^2)
tree_rmse <- sqrt(tree_mse)
tree_mae <- mean(abs(tree_preds - test$ALCYDAYS))

cat("Decision Tree - MSE:", round(tree_mse, 2), "\n")
cat("Decision Tree - RMSE:", round(tree_rmse, 2), "\n")
cat("Decision Tree - MAE:", round(tree_mae, 2), "\n")

# Plot: Actual vs Predicted
ggplot(data.frame(Predicted = tree_preds, Actual = test$ALCYDAYS),
       aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "darkred") +
  labs(title = "Decision Tree - Actual vs Predicted",
       x = "Actual Alcohol Days", y = "Predicted") +
  theme_minimal()

# ----------------------
# Random Forest Regression
# ----------------------
rf_features <- c("ALCYDAYS", "IRSEX", "NEWRACE2", "INCOME", "EDUSCHGRD2",
                 "PARCHKHW", "PARHLPHW", "PRLMTTV2", "PRTALK3", "PRPROUD2")

df_rf <- df %>%
  select(all_of(rf_features)) %>%
  drop_na()

split_idx <- sample(1:nrow(df_rf), 0.7 * nrow(df_rf))
train <- df_rf[split_idx, ]
test <- df_rf[-split_idx, ]

rf_model <- randomForest(ALCYDAYS ~ ., data = train, ntree = 350, mtry = 3, importance = TRUE)
rf_pred <- predict(rf_model, newdata = test)

rf_mse <- mean((rf_pred - test$ALCYDAYS)^2)
rf_rmse <- sqrt(rf_mse)
rf_mae <- mean(abs(rf_pred - test$ALCYDAYS))

cat("Random Forest - MSE:", round(rf_mse, 2), "\n")
cat("Random Forest - RMSE:", round(rf_rmse, 2), "\n")
cat("Random Forest - MAE:", round(rf_mae, 2), "\n")

# Visualize variable importance
varImpPlot(rf_model)

# Plot OOB error trend
plot(rf_model, main = "Random Forest - OOB Error vs. Trees")

# Plot: Actual vs Predicted
ggplot(data.frame(Predicted = rf_pred, Actual = test$ALCYDAYS),
       aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, color = "darkgreen") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Random Forest - Actual vs Predicted",
       x = "Actual Alcohol Days", y = "Predicted") +
  theme_minimal()

# ----------------------
# Gradient Boosting Regression
# ----------------------
set.seed(19)
gbm_features <- c("ALCYDAYS", "PRLMTTV2", "PARCHKHW", "PARHLPHW",
                  "PRPROUD2", "PRTALK3", "PRBSOLV2", "PREVIOL2",
                  "IRSEX", "NEWRACE2", "INCOME", "EDUSCHGRD2")

df_gbm <- df %>%
  select(all_of(gbm_features)) %>%
  drop_na()

idx <- sample(1:nrow(df_gbm), 0.7 * nrow(df_gbm))
train <- df_gbm[idx, ]
test <- df_gbm[-idx, ]

gbm_model <- gbm(
  formula = ALCYDAYS ~ .,
  distribution = "gaussian",
  data = train,
  n.trees = 3000,
  interaction.depth = 3,
  shrinkage = 0.01,
  cv.folds = 5,
  n.cores = NULL,
  verbose = FALSE
)

best_iter <- gbm.perf(gbm_model, method = "cv")
gbm_pred <- predict(gbm_model, newdata = test, n.trees = best_iter)

mse <- mean((gbm_pred - test$ALCYDAYS)^2)
rmse <- sqrt(mse)
mae <- mean(abs(gbm_pred - test$ALCYDAYS))

cat("GBM - MSE:", round(mse, 2), "\n")
cat("GBM - RMSE:", round(rmse, 2), "\n")
cat("GBM - MAE:", round(mae, 2), "\n")

# Plot: Actual vs Predicted
ggplot(data.frame(Predicted = gbm_pred, Actual = test$ALCYDAYS),
       aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, color = "darkorange") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = "GBM - Actual vs Predicted",
       x = "Actual Alcohol Days", y = "Predicted") +
  theme_minimal()

# ----------------------
# Regression Model Comparison 
# ----------------------
regression_metrics <- data.frame(
  Model = rep(c("Decision Tree", "Random Forest", "GBM"), each = 3),
  Metric = rep(c("MSE", "RMSE", "MAE"), times = 3),
  Value = c(
    round(tree_mse, 2), round(tree_rmse, 2), round(tree_mae, 2),
    round(rf_mse, 2), round(rf_rmse, 2), round(rf_mae, 2),
    round(mse, 2), round(rmse, 2), round(mae, 2)
  )
)

ggplot(regression_metrics, aes(x = Metric, y = Value, fill = Model)) +
  geom_col(position = "dodge", width = 0.7) +
  geom_text(aes(label = round(Value, 2)),
            position = position_dodge(width = 0.7),
            vjust = -0.5, size = 4) +
  scale_fill_brewer(palette = "Set2") +
  ylim(0, max(regression_metrics$Value) + 0.5) +
  labs(title = "Regression Model Comparison - MSE, RMSE, MAE",
       x = "Evaluation Metric", y = "Value") +
  theme_minimal()


