# Problem Set 3: Claims modelling exercise

## To install the package

```
conda env create -f environment.yml
conda activate ps3

pre-commit install
pip install --no-build-isolation -e .
```

## Objective

The goal is to improve the risk model presented in the [GLM Tutorial: Poisson, Gamma, and Tweedie with French Motor](https://glum.readthedocs.io/en/latest/tutorials/glm_french_motor_tutorial/glm_french_motor.html#) and learn how to work with model pipelines.

## Tasks

### 1. **Load the Data**

* Use the function `load_transform` as defined in the repository to load the data.

### 2. **Adjust Train/Test Split**

* Import the splitting function and assign the **sample column** to split the dataset into training and testing sets.

### 3. **Initial Benchmark - Train the Tweedie Model for Pure Premium**

* Train a baseline **Tweedie model** for predicting **pure premium** (claim amount divided by exposure).
* **Why divide by exposure?**

  * We divide by **exposure** to account for policies with different active durations. For example, one person could have a policy for 16 years, while another could have it for 8 years. Without dividing by exposure, the claims for the two customers would be difficult to compare fairly. The aim is to compare **risk fairly**.

### 4. **Improving the Parametric Model**

* **Add Flexibility to the Model**:

  * For the variables **BonusMalus** and **VehPower**, we initially treated them as **linear terms**. Now, we will model them using **Polynomials** or **Splines** to allow for more flexibility in capturing their relationship with the target variable.
* **Deviance on Train and Test Set**:

  * Observe the changes in **deviance** on the training and test sets after adding splines. If the training deviance drops significantly while the test deviance increases, it might indicate overfitting.

| Model          | Train Dev | Test Dev |
| -------------- | --------- | -------- |
| GLM1 (linear)  | 74.096    | 72.420   |
| GLM2 (splines) | 73.706    | 72.252   |

* **Conclusion**: Adding splines improved the deviance on both the training and test data. The **decrease in deviance** suggests better fit, but since both training and test deviance decreased, it does not indicate overfitting.

* **Checking for Overfitting**:

  * We can check for overfitting by looking at the deviance values for training and testing sets. If there's a large gap between them (training loss much lower than testing loss), it suggests overfitting. We can also monitor the **instability of predictions** on the test set.

### 5. **Using LGBM Regressor for Further Improvement**

* **LGBM Regressor** is a powerful estimator, and we aim to use it for improving the model.

* **Why These Parameters for LGBM?**

| Parameter                    | Explanation                                            |
| ---------------------------- | ------------------------------------------------------ |
| `objective="tweedie"`        | The objective function for **GLM-type regression**.    |
| `tweedie_variance_power=1.5` | Matches the variance used in **GLM Tweedie**.          |
| `learning_rate=0.05`         | Ensures **stability** in model convergence.            |
| `min_data_in_leaf=200`       | Reduces **overfitting** by controlling leaf size.      |
| `num_leaves=31`              | Prevents overly **complex trees** and **overfitting**. |
| `n_estimators=500`           | Sufficient trees for smooth learning.                  |

#### Step 1: Define the Modelling Pipeline

```python
lgbm_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),  # Preprocessing step
        (
            "estimate",
            LGBMRegressor(
                objective="tweedie",  # GLM-type regression
                tweedie_variance_power=1.5,
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                min_data_in_leaf=200,
                subsample=0.8,
                colsample_bytree=0.8,
            ),
        ),
    ]
)
```

#### Step 2: Fit the Model

```python
model_pipeline.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)
```

* The model will now be trained with **sample weights** (`w_train_t`) that reflect the risk associated with each policyholder.

#### Step 3: Hyperparameter Tuning with GridSearchCV

```python
param_grid = {
    "estimate__learning_rate": [0.1, 0.05, 0.01],
    "estimate__n_estimators": [300, 500, 1000],
}

cv = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring="neg_mean_poisson_deviance",  # Use Poisson deviance as the metric
    cv=3,  # 3-fold cross-validation
    n_jobs=-1,  # Use all available cores
    verbose=1,  # Show detailed progress
)

cv.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)
```

* This will search for the best combination of `learning_rate` and `n_estimators` to minimize **Poisson deviance** on the training data.

* **Best Parameters**:

```python
# Access the best parameters found by GridSearchCV
print("Best hyperparameters found by GridSearchCV:")
print(cv.best_params_)
```

The output might look like:

```
Best hyperparameters found by GridSearchCV:
{
    'estimate__learning_rate': 0.01,
    'estimate__n_estimators': 300
}
```

### 6. **Optional: Train a LGBM Frequency and Severity Model**

* Once you have the baseline LGBM model, you can further improve by splitting the model into **frequency** and **severity** components (i.e., separate models for predicting the number of claims and the claim amounts).

* **Compare the performance** of this model to the other models you’ve tried.

### Tips:

* Use `dir(object)` to explore all the attributes of a model and check what’s been stored, especially the best hyperparameters found by the grid search.

---

## Conclusion:

* The **LGBM Regressor** model provides a solid foundation for predicting claims.
* Through **hyperparameter tuning**, we were able to optimize the learning rate and number of estimators.
* Adding **splines** improved the model's performance and flexibility without overfitting.
* **Overfitting checks** and **cross-validation** ensure that the model generalizes well.

---
