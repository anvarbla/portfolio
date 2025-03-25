# **Objective:** 

The objective of this project is to develop the best machine learning model to predict the final cost of manufacturing an airplane. 

# **Datasets:**
## **Airplane_price_dataset:**

* **Categorical Columns:**  \
‘model’ - Different models of airplanes based on the manufacturing company (e.g., Airbus, Boeing).
‘sales region’ - The region where the airplane was manufactured (e.g., North America, South America, Europe, Asia, Africa, Oceania).
‘engine type’ - Type of engine used in the airplane.

* **Numerical Columns:**  \
‘production year’ - The year the airplane was produced.
‘capacity’ - The number of passengers the airplane can accommodate.
‘range (km)’- The maximum distance (in kilometers) the airplane can fly without refueling.
‘fuel consumption’- Fuel consumption (in liters) per hour.
‘maintenance cost per hour’ - The cost of maintenance per hour of operation.
‘number of motors’- The total number of engines on the airplane.


# **Categorical columns Analysis:**

To determine which categorical variables to include in our machine learning model, we performed three hypothesis tests. We only included those variables whose means (in terms of prices) are significantly different at a 95% confidence level.

**Result**: We found sufficient evidence to conclude that the means in terms of prices for model and engine type are significantly different, and we included these variables in our model.

# **Numerical columns Analysis:**

To select the numerical variables for our machine learning model, we conducted a correlation analysis to identify features that are highly correlated with the price. We encountered multicollinearity among some features, so we performed a Variance Inflation Factor (VIF) test to address this issue.

**Result**: We retained the following variables: fuel consumption, number of engines, and range (km), all of which had a VIF score of less than 8.

**Final features included in our model**: model, engine type, fuel consumption, number of engines, and range (km).

# **Machine learning model (first approach):**

In our initial approach, we trained and tested several models, obtaining the following accuracy results:

**K-Nearest Neighbors (KNN)**: 0.85
**Decision Tree**: 0.84
**Bagging and Pasting**: 0.86
**Random Forest**: 0.86
**AdaBoosting**: 0.83
**Gradient Boosting**: 0.82

This indicates that our model can accurately predict approximately 85% of the prices from our dataset of 12,000 samples. However, we observed that the Random Forest, AdaBoosting, and Gradient Boosting models were prone to overfitting, which could lead to issues in prediction accuracy.

# **Hyperparameters and final model validation**

To enhance the accuracy of our model, we employed two different methods for hyperparameter optimization: **Random Search** and **Grid Search**.

By setting the number of estimators to 100 and the maximum depth to 5 for both the **Random Forest** and **Gradient Boosting** methods, we achieved an improved accuracy of **0.87**, which is the best result obtained. It is also important to note that this combination helped reduce overfitting.
