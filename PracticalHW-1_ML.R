# Load libraries
library(tidyverse)
library(tree)
library(randomForest)
library(gbm)
library(caret)
library(dplyr)

# Load processed dataset
load("/Users/pavan/Downloads/youth_data.Rdata")
#BINARY CLASSIFICATION
#QUESTION : Can we predict whether a youth is currently attending school based on their substance use behavior?

# Recreate target variable
df$EDUSKPCOM <- as.numeric(as.character(df$EDUSKPCOM))
df$SKIPPED_SCHOOL <- ifelse(df$EDUSKPCOM >= 1 & df$EDUSKPCOM <= 30, 1,
                            ifelse(df$EDUSKPCOM == 0, 0, NA))

# Select substance + youth + demographics
expanded_df <- df %>%
  select(SKIPPED_SCHOOL,
         IRALCFM, IRMJFM, IRCIGFM,
         IRMJAGE, IRALCAGE, IRCIGAGE,
         MRJFLAG, ALCFLAG, TOBFLAG,
         all_of(youth_experience_cols),
         all_of(demographic_cols)) %>%
  select(-EDUSKPCOM)  

# Handle 991s in numeric columns
num_cols <- c("IRALCFM", "IRMJFM", "IRCIGFM", "IRMJAGE", "IRALCAGE", "IRCIGAGE")
expanded_df[num_cols] <- lapply(expanded_df[num_cols], function(x) {
  x <- as.numeric(as.character(x))
  na_if(x, 991)
})

# Convert SKIPPED_SCHOOL + flags to factors
expanded_df$SKIPPED_SCHOOL <- factor(expanded_df$SKIPPED_SCHOOL)
expanded_df$MRJFLAG <- factor(expanded_df$MRJFLAG)
expanded_df$ALCFLAG <- factor(expanded_df$ALCFLAG)
expanded_df$TOBFLAG <- factor(expanded_df$TOBFLAG)

# Drop rows with NA
expanded_df <- tidyr::drop_na(expanded_df)

# Confirm dimensions
glimpse(expanded_df)

# Load library
library(randomForest)

# Set seed for reproducibility
set.seed(19)

# 'SKIPPED' is the binary target (0 = No, 1 = Yes)
features <- setdiff(names(binary_df), "SKIPPED")

# Train-test split
idx <- sample(1:nrow(binary_df), 0.7 * nrow(binary_df))
train <- binary_df[idx, ]
test <- binary_df[-idx, ]

# Bagging Model: mtry = number of predictors
bag_model <- randomForest(
  SKIPPED ~ ., 
  data = train, 
  mtry = length(features), 
  ntree = 100, 
  importance = TRUE
)

# Prediction
probs <- predict(bag_model, newdata = test, type = "prob")[, 2]
bag_pred_thresh <- ifelse(probs > 0.4, 1, 0)

# Evaluation
conf_matrix <- table(Predicted = bag_pred_thresh, Actual = test$SKIPPED)
print(conf_matrix)
cat("Bagging Test Accuracy:", round(mean(bag_pred_thresh == test$SKIPPED), 3), "\n")
# Variable importance
varImpPlot(bag_model)
plot(bag_model)

#decision tree
set.seed(19)

sample_index <- sample(seq_len(nrow(expanded_df)), size = 0.7 * nrow(expanded_df))
train_data <- expanded_df[sample_index, ]
test_data <- expanded_df[-sample_index, ]

#  Fit decision tree
tree_model <- tree(SKIPPED_SCHOOL ~ ., data = train_data)

# Plot tree
plot(tree_model)
text(tree_model, pretty = 0)

# Tree summary
summary(tree_model)

# Predict on test data
tree_pred_test <- predict(tree_model, newdata = test_data, type = "class")

# Confusion matrix
conf_matrix_test <- table(Predicted = tree_pred_test, Actual = test_data$SKIPPED_SCHOOL)
print(conf_matrix_test)

# Accuracy
test_accuracy <- sum(diag(conf_matrix_test)) / sum(conf_matrix_test)
paste("Test Accuracy:", round(test_accuracy, 3))

summary(tree_model)

# Random Forest - Binary Classification
set.seed(19)

# Fit the random forest model with selected mtry
rf_model <- randomForest(
  SKIPPED_SCHOOL ~ .,
  data = train_data,
  ntree = 500,
  mtry = 4,
  importance = TRUE
)

# Model summary
print(rf_model)

# Predict on test data
rf_pred_test <- predict(rf_model, newdata = test_data)

# Confusion matrix
rf_conf_test <- table(Predicted = rf_pred_test, Actual = test_data$SKIPPED_SCHOOL)
print(rf_conf_test)

# Accuracy calculation
rf_test_accuracy <- sum(diag(rf_conf_test)) / sum(rf_conf_test)
cat("Random Forest Test Accuracy (mtry = 4):", round(rf_test_accuracy, 3), "\n")

# Plot variable importance
varImpPlot(rf_model, main = "Random Forest Variable Importance")

#GBM
# Copy train and test for GBM
train_gbm <- train_data
test_gbm <- test_data

# Convert SKIPPED_SCHOOL to numeric
train_gbm$SKIPPED_SCHOOL <- as.numeric(as.character(train_gbm$SKIPPED_SCHOOL))
test_gbm$SKIPPED_SCHOOL <- as.numeric(as.character(test_gbm$SKIPPED_SCHOOL))

set.seed(19)
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

# Find best number of trees via CV
best_iter <- gbm.perf(gbm_model, method = "cv")
cat("Best iteration:", best_iter, "\n")

# Predict probabilities on test set
gbm_probs <- predict(gbm_model, newdata = test_gbm, n.trees = best_iter, type = "response")

# Use 0.4 threshold to improve recall
gbm_preds <- ifelse(gbm_probs > 0.4, 1, 0)

# Confusion matrix
gbm_conf <- table(Predicted = gbm_preds, Actual = test_gbm$SKIPPED_SCHOOL)
print(gbm_conf)

# Accuracy
gbm_accuracy <- sum(diag(gbm_conf)) / sum(gbm_conf)
paste("GBM Test Accuracy:", round(gbm_accuracy, 3))
summary(gbm_model, n.trees = best_iter, main = "GBM Variable Importance")

#MULTICLASS CLASSIFICATION
#Can we classify the frequency of marijuana use among youth into None, Occasional, and Frequent groups based on their home environment and peer influence?
# Load required libraries
library(tidyverse)
library(tree)

# Set seed
set.seed(19)

# Create SKIP_LEVEL from MRJMDAYS
df$SKIP_LEVEL <- case_when(
  df$MRJMDAYS == 5 ~ "None",
  df$MRJMDAYS %in% c(3, 4) ~ "Occasional",
  df$MRJMDAYS %in% c(1, 2) ~ "Frequent"
)
df$SKIP_LEVEL <- factor(df$SKIP_LEVEL)

# Select features (home environment + peer influence)
features <- c("SKIP_LEVEL",
              "PARCHKHW", "PARHLPHW", "PRPROUD2", "PRPKCIG2",
              "PRTALK3", "PRBSOLV2", "PREVIOL2",
              "YFLMJMO", "YFLTMRJ2", "FRDMJMON", "PRMJEVR2")

final_df <- df %>%
  select(all_of(features)) %>%
  drop_na()

# Downsample to fix class imbalance
min_n <- min(table(final_df$SKIP_LEVEL))
final_df <- final_df %>%
  group_by(SKIP_LEVEL) %>%
  slice_sample(n = min_n) %>%
  ungroup()

# Train-test split
idx <- sample(1:nrow(final_df), 0.7 * nrow(final_df))
train <- final_df[idx, ]
test <- final_df[-idx, ]

# Fit Decision Tree
tree_model <- tree(SKIP_LEVEL ~ ., data = train)

# Plot Tree
plot(tree_model)
text(tree_model, pretty = 0)

# Predict and Evaluate
tree_pred <- predict(tree_model, newdata = test, type = "class")
conf_matrix <- table(Predicted = tree_pred, Actual = test$SKIP_LEVEL)
print(conf_matrix)

#  Accuracy
acc <- mean(tree_pred == test$SKIP_LEVEL)
cat("Decision Tree Test Accuracy:", round(acc, 3), "\n")

# Load required libraries
library(dplyr)
library(randomForest)
library(caret)

# Convert MRJMDAYS to numeric and drop NA
df$MRJMDAYS <- as.numeric(as.character(df$MRJMDAYS))
df <- df %>% filter(!is.na(MRJMDAYS) & MRJMDAYS <= 5)

# Rebucket MRJMDAYS into 3 groups
df$SKIP_LEVEL <- case_when(
  df$MRJMDAYS == 5 ~ "None",
  df$MRJMDAYS %in% c(3, 4) ~ "Occasional",
  df$MRJMDAYS %in% c(1, 2) ~ "Frequent"
)
df$SKIP_LEVEL <- factor(df$SKIP_LEVEL, levels = c("Frequent", "None", "Occasional"))

# Select features related to home environment and peer influence
home_peer_cols <- c(
  "PARCHKHW", "PARHLPHW", "PRPROUD2", "PRPKCIG2", "PRTALK3", "PRBSOLV2",
  "PREVIOL2", "FRDMJMON", "PRMJEVR2", "YFLMJMO", "YFLTMRJ2"
)

# Build final dataset
multi_df <- df %>%
  select(SKIP_LEVEL, all_of(home_peer_cols)) %>%
  drop_na()

# Downsample to balance classes
set.seed(19)
down_df <- multi_df %>%
  group_by(SKIP_LEVEL) %>%
  sample_n(min(table(multi_df$SKIP_LEVEL))) %>%
  ungroup()

# Split into train and test
set.seed(19)
idx <- sample(1:nrow(down_df), 0.7 * nrow(down_df))
train <- down_df[idx, ]
test <- down_df[-idx, ]

# Fit Random Forest
rf_model <- randomForest(SKIP_LEVEL ~ ., data = train, ntree = 200, mtry = 4)

# Predict and evaluate
rf_pred <- predict(rf_model, test, type = "class")
conf_matrix <- confusionMatrix(rf_pred, test$SKIP_LEVEL)

# Print results
print(conf_matrix)
cat("Test Accuracy:", round(mean(rf_pred == test$SKIP_LEVEL), 3), "\n")

# Variable importance
varImpPlot(rf_model)

# Load required libraries
library(gbm)
library(caret)

set.seed(19)

# Convert SKIP_LEVEL to numeric for multinomial classification
gbm_df <- final_df
gbm_df$SKIP_LEVEL_NUM <- as.numeric(as.factor(gbm_df$SKIP_LEVEL))

# Train-test split
idx <- sample(1:nrow(gbm_df), 0.7 * nrow(gbm_df))
train_gbm <- gbm_df[idx, ]
test_gbm <- gbm_df[-idx, ]

# Fit GBM model
gbm_model <- gbm(
  formula = SKIP_LEVEL_NUM ~ . -SKIP_LEVEL,
  data = train_gbm,
  distribution = "multinomial",
  n.trees = 3000,
  interaction.depth = 3,
  shrinkage = 0.01,
  cv.folds = 5,
  verbose = FALSE
)

# Best iteration
best_iter <- gbm.perf(gbm_model, method = "cv")

# Predict probabilities on test set
gbm_probs <- predict(gbm_model, newdata = test_gbm, n.trees = best_iter, type = "response")

# Convert to predicted class
gbm_pred_numeric <- apply(gbm_probs, 1, which.max)

# Convert numeric prediction to factor to match SKIP_LEVEL
true_levels <- levels(final_df$SKIP_LEVEL)
predicted_factor <- factor(gbm_pred_numeric, labels = true_levels)
actual_factor <- factor(test_gbm$SKIP_LEVEL_NUM, labels = true_levels)

# Confusion Matrix
conf_matrix <- confusionMatrix(predicted_factor, actual_factor)
print(conf_matrix)

# Accuracy
cat("GBM Test Accuracy:", round(mean(predicted_factor == actual_factor), 3), "\n")

#REGRESSION

#Can we predict how many days a youth drank alcohol in the past year based on their demographics and parental involvement?
# Set seed for reproducibility
set.seed(19)

# Define new feature set to avoid over-splitting
regression_features <- c(
  "ALCYDAYS",        
  "FRDADLY2",        
  "PARCHKHW",        
  "PRBSOLV2",        
  "PRPROUD2",        
  "YTHACT2",         
  "NEWRACE2",       
  "IRSEX",           
  "INCOME",          
  "EDUSCHGRD2"       
)

# Filter and clean data
df_reg <- df %>%
  select(all_of(regression_features)) %>%
  drop_na()

# Train-test split
idx <- sample(1:nrow(df_reg), 0.7 * nrow(df_reg))
train <- df_reg[idx, ]
test <- df_reg[-idx, ]

# Fit regression tree with split control
tree_model <- tree(
  ALCYDAYS ~ .,
  data = train,
  control = tree.control(
    nobs = nrow(train),
    mincut = 15,
    minsize = 30,
    mindev = 0.01
  )
)

# Plot tree
plot(tree_model)
text(tree_model, pretty = 0)

# Predict and calculate MSE
tree_preds <- predict(tree_model, newdata = test)
tree_mse <- mean((tree_preds - test$ALCYDAYS)^2)
cat("Regression Tree MSE:", round(tree_mse, 3), "\n")

#Random Forest

# Load required library
library(randomForest)

# Set seed for reproducibility
set.seed(19)

# Select relevant variables
regression_vars <- c("ALCYDAYS", "IRSEX", "NEWRACE2", "INCOME", "EDUSCHGRD2", 
                     "PARCHKHW", "PARHLPHW", "PRLMTTV2", "PRTALK3", "PRPROUD2")

# Create a cleaned dataframe
df_rf <- df %>%
  dplyr::select(all_of(regression_vars)) %>%
  tidyr::drop_na()

# Train-test split
split_idx <- sample(1:nrow(df_rf), 0.7 * nrow(df_rf))
train <- df_rf[split_idx, ]
test <- df_rf[-split_idx, ]

# Fit Random Forest Regression model
rf_model <- randomForest(ALCYDAYS ~ ., data = train, ntree = 350, mtry = 3, importance = TRUE)

# Predict on test data
rf_pred <- predict(rf_model, newdata = test)

# Compute Mean Squared Error
rf_mse <- mean((rf_pred - test$ALCYDAYS)^2)
cat("Random Forest MSE:", round(rf_mse, 2), "\n")

# Plot variable importance
varImpPlot(rf_model)

# Optional: Plot OOB error by tree count
plot(rf_model, main = "Random Forest - OOB Error vs. Trees")
mae <- mean(abs(rf_pred - test$ALCYDAYS))
cat("MAE:", round(mae, 2), "\n")

# Load library
library(gbm)

# Set seed for reproducibility
set.seed(19)

# Select relevant features and target
regression_features <- c("ALCYDAYS", "PRLMTTV2", "PARCHKHW", "PARHLPHW", 
                         "PRPROUD2", "PRTALK3", "PRBSOLV2", "PREVIOL2", 
                         "IRSEX", "NEWRACE2", "INCOME", "EDUSCHGRD2")

df_reg <- df %>%
  select(all_of(regression_features)) %>%
  drop_na()

# Train-test split
idx <- sample(1:nrow(df_reg), 0.7 * nrow(df_reg))
train <- df_reg[idx, ]
test <- df_reg[-idx, ]

# Fit GBM model
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

# Optimal number of trees based on CV
best_iter <- gbm.perf(gbm_model, method = "cv")

# Predict on test data
gbm_pred <- predict(gbm_model, newdata = test, n.trees = best_iter)

# Evaluation metrics
mse <- mean((gbm_pred - test$ALCYDAYS)^2)
mae <- mean(abs(gbm_pred - test$ALCYDAYS))
rmse <- sqrt(mse)

cat("GBM MSE:", round(mse, 2), "\n")
cat("GBM MAE:", round(mae, 2), "\n")
cat("GBM RMSE:", round(rmse, 2), "\n")

 
