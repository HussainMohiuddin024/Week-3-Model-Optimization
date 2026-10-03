Week 3: Model Optimization and Unsupervised Learning

This project is part of Week 3 of the Applied AI course. The main goal was to understand why a single train/test split can be misleading and how cross-validation, hyperparameter tuning, early stopping, clustering, and PCA can improve model evaluation and interpretation.

Project Objectives

* Evaluate the effect of different random train/test splits.
* Compare Logistic Regression, Random Forest, and XGBoost using 5-fold cross-validation.
* Tune machine learning models using Grid Search and Randomized Search.
* Use early stopping with XGBoost.
* Apply K-means clustering to identify customer segments.
* Use PCA to understand feature variance and relationships.
* Select a final model using cross-validation and evaluate it once on the locked test set.

Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains customer information such as tenure, monthly charges, total charges, services, contract information, and churn status.

The test set was created at the beginning of the experiment and was kept untouched until the final evaluation.

Split-to-Split Experiment

The same Logistic Regression model was trained using different random train/test splits.

* Minimum accuracy: 0.780
* Maximum accuracy: 0.828
* Standard deviation: 0.0124
* Theoretical standard error: 0.0107
* Approximate 95% interval: ±0.021

This experiment showed that model accuracy can change noticeably simply because the random split changes. Therefore, relying on one train/test split can lead to an unstable conclusion.

5-Fold Cross-Validation

The models were compared using 5-fold cross-validation with ROC-AUC as the evaluation metric.

Model	Mean CV AUC	Std
Logistic Regression (tuned C)	0.8464	0.0129
Random Forest (random search)	0.8464	0.0114
XGBoost (tuned)	0.8502	0.0117

XGBoost achieved the highest mean cross-validation AUC of 0.8502. The results also show that the difference between the three models was relatively small, which makes cross-validation important for making a more reliable comparison.

XGBoost Optimization

XGBoost was first evaluated using early stopping.

* scale_pos_weight: 2.77
* Best number of trees during early stopping: 247
* Validation AUC: 0.8541

The validation loss improved initially and then stopped improving significantly. Early stopping around 247 boosting rounds helped prevent unnecessary additional trees.

Tuned XGBoost

RandomizedSearchCV was used with 30 random configurations and 5-fold cross-validation.

Best parameters:

colsample_bytree = 0.561
learning_rate = 0.034
max_depth = 2
min_child_weight = 1
n_estimators = 476
reg_lambda = 1.974
subsample = 0.6

Tuned XGBoost CV AUC:

0.8502

Customer Segmentation with K-means

K-means clustering was applied using customer features including:

* Tenure
* MonthlyCharges
* TotalCharges
* Number of services

The features were standardized before clustering because they have different numerical scales.

The elbow method and silhouette analysis were used to investigate different values of K. Based on the clustering structure and business interpretability, K = 4 was selected.

The four customer segments represent different combinations of customer tenure, spending, and service usage.

PCA Analysis

PCA was applied to the encoded feature set to understand how much variance could be represented by fewer dimensions.

* Total components: 30
* Components required for 90% variance: 15

Therefore, 15 of 30 components explain 90% of the variance.

The PCA loading analysis also showed several highly related service-related dummy variables. The PC1/PC2 visualization showed considerable overlap between churned and non-churned customers, indicating that churn is not separated cleanly in just two principal components.

Final Model Evaluation

The final model was selected using the cross-validation results.

Final model: XGBoost (tuned)

The test set was used only once for the final evaluation.

* Test ROC-AUC: 0.8483
* Recall: 0.521
* Precision: 0.659
* Cross-validation mean AUC: 0.8502

The test AUC was slightly lower than the cross-validation mean by 0.0019, showing a close agreement between cross-validation performance and final test performance.

The final model was saved as:

churn_model.joblib

Key Lessons

The biggest lesson from this experiment was that a single accuracy score is not enough to judge a machine learning model. Different random splits can produce different results, so cross-validation gives a more stable basis for comparing models.

I also learned that hyperparameter tuning, early stopping, feature scaling, clustering, and PCA can provide useful information beyond simply training a model and checking its accuracy.

Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Joblib
* Kaggle Notebooks
* GitHub

Project Deliverables

* Kaggle notebook: Week 3 - Model Optimization
* Model file: churn_model.joblib
* GitHub README
* Model comparison using 5-fold cross-validation
* XGBoost tuning and early-stopping analysis
* K-means customer segmentation
* PCA analysis
* Final test-set evaluation

Conclusion

This project demonstrated a complete machine learning workflow from model evaluation and optimization to unsupervised learning. The results showed that XGBoost achieved a CV AUC of 0.8502 and a final test AUC of 0.8483, while K-means and PCA provided additional insight into customer structure and feature relationships.
