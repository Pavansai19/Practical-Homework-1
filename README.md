# Practical Homework 1 - Decision Trees and Ensemble Models on Youth Drug Use (NSDUH 2023)

## Overview
This project investigates youth behavior patterns and substance use using the NSDUH 2023 dataset. We explored three machine learning tasks using decision trees and ensemble models:

- **Binary Classification**: Predicting whether a youth skipped school recently
- **Multiclass Classification**: Categorizing frequency of marijuana use
- **Regression**: Estimating how many days a youth drank alcohol in the past year

All models were implemented in **R**, with a focus on clean code, ethical modeling practices, and robust evaluation.

---

## Dataset
We used the **National Survey on Drug Use and Health (NSDUH) 2023**, focusing on:
- Youth substance use (alcohol, marijuana, tobacco)
- Peer influence and home environment variables
- Demographics (age, sex, race, income, education)

---

## Tasks & Methodology

### 1. Binary Classification
**Goal**: Can we predict if a youth skipped school recently?

- **Models Used**: Decision Tree, Bagging, Random Forest, GBM
- **Thresholding**: Custom probability threshold (0.4) for better recall
- **Evaluation**: Accuracy, Precision, Recall, F1, Confusion Matrix Heatmaps
- **Findings**:
  - Bagging performed best (F1 Score: 0.766)
  - Pruned Decision Tree underperformed
  - Random Forest balanced performance with interpretability

### 2. Multiclass Classification
**Goal**: Can we classify marijuana use into None, Occasional, and Frequent?

- **Models Used**: Decision Tree, Random Forest, GBM
- **Class Imbalance**: Manually downsampled each class to ensure fairness
- **Evaluation**: Class-wise F1 Scores, Macro-F1, Confusion Matrix Heatmaps
- **Findings**:
  - GBM performed best overall (Macro F1 Score: 0.586)
  - Tree models had difficulty identifying the 'Frequent' class

### 3. Regression
**Goal**: Predict alcohol consumption days in the past year

- **Models Used**: Decision Tree, Random Forest, GBM
- **Evaluation**: MSE, RMSE, MAE, Actual vs Predicted Plots
- **Findings**:
  - All models performed similarly
  - Random Forest had the lowest error (MAE: 1.17)
  - Feature importance and prediction trends were visualized

---

## Ethical Modeling Approach
- **Avoided data leakage** by not using target-leaking variables
- **Handled class imbalance** via controlled downsampling
- **Used consistent seed (`set.seed(19)`)** across all tasks for reproducibility
- **Evaluated models beyond accuracy**, using class-specific metrics

---

## Deliverables
- R script implementing all tasks and models (`PracticalHW-1_ML.R`)
- All plots saved as PNGs (confusion matrices, F1 comparisons, regression scatter plots)
- Final `.Rmd` 

---

## Results Summary
| Task                   | Best Model     | Metric Used | Performance       |
|------------------------|----------------|--------------|--------------------|
| Binary Classification | Bagging        | F1 Score     | **0.766**         |
| Multiclass Classification | GBM         | Macro F1     | **0.586**         |
| Regression            | Random Forest  | MAE          | **1.17** days     |

---

## Acknowledgements
This project was completed as part of a Machine Learning practical assignment. All work adheres to ethical data science practices and was conducted in R.


---

Feel free to explore the code and plots, and reach out with any questions!
