
 Model Setup and Implementation
We trained the following models for each task:
Binary Classification – “Did the student skip school?”
	Models Used:
	•	Decision Tree
	•	Random Forest
	•	Bagging (with threshold tuning)
	•	Gradient Boosting Machine (GBM)
	Hyperparameters:
	•	Random Forest: ntree = 500, mtry = 4
	•	GBM: n.trees = 5000, shrinkage = 0.01, interaction.depth = 3
	•	Bagging: mtry = number of predictors, tested thresholds: 0.3, 0.4, 0.5
Multi-Class Classification – “How frequently does a student use marijuana?”
	Target: Recoded MRJMDAYS into 3 classes.
	Features: Chosen from home environment and peer influence domains only.
	Models Used:
	•	Decision Tree
	•	Random Forest
	•	GBM (multinomial distribution)
	Class Imbalance Handling:
	•	We applied downsampling to ensure equal representation across the three marijuana usage categories.
	•	Also tried restricted model complexity to reduce overfitting.
	•	Explored class weight adjustments but finally stuck with sampling.
Regression – “How many days per year does a student consume alcohol?”
	Target: ALCYDAYS
	Features: Parental behavior, gender, race, income, education.
	Models Used:
	•	Decision Tree
	•	Random Forest
	•	GBM (Gaussian loss function)
	Hyperparameter Tuning:
	•	For RF: ntree = 350, mtry = 3
	•	For GBM: n.trees = 3000, shrinkage = 0.01, cv.folds = 5
Model Evaluation & Validation
Each model was trained using a 70-30 train-test split. For evaluation:
	Classification Models:
	•	Confusion matrix
	•	Accuracy, Precision, Recall, F1-Score
	•	Class-specific sensitivity and specificity
	Regression Models:
	•	Mean Squared Error (MSE)
	•	Mean Absolute Error (MAE)
	•	Root Mean Squared Error (RMSE)
We also generated variable importance plots to interpret which features most influenced each model.

 

Key Refinements and Challenges
To prevent data leakage, we carefully excluded any predictors directly related to the target.
Downsampling was preferred over weighting due to skewed class distributions in multi-class classification.
Threshold tuning in bagging and GBM significantly improved recall in the binary classification task.
We encountered overfitting in early stages, which was handled by limiting model complexity and adjusting sampling.
We monitored the OOB error vs. ntree plots to choose the optimal number of trees for ensemble methods.
