# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
import dalex as dx

from ps3.data import create_sample_split, load_transform
from ps3.evaluation import evaluate_predictions

# %%
# load data
df = load_transform()

# %%
# Train benchmark tweedie model. This is entirely based on the glum tutorial.
weight = df["Exposure"].values
df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
y = df["PurePremium"]
# TODO: Why do you think, we divide by exposure here to arrive at
# our outcome variable?
# Because we would be looking at policies with different active durations.
# One person could have a policy of 16 years, and one for 8.
# The claim amounts would be different because the timespan is different.
# We compare risk fairly

df = create_sample_split(df, 'IDpol')
train = np.where(df["sample"] == "train")
test = np.where(df["sample"] == "test")
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()

categoricals = ["VehBrand", "VehGas", "Region", "Area",
                "DrivAge", "VehAge", "VehPower"]

predictors = categoricals + ["BonusMalus", "Density"]
glm_categorizer = Categorizer(columns=categoricals)

X_train_t = glm_categorizer.fit_transform(df[predictors].iloc[train])
X_test_t = glm_categorizer.transform(df[predictors].iloc[test])
y_train_t, y_test_t = y.iloc[train], y.iloc[test]
w_train_t, w_test_t = weight[train], weight[test]

TweedieDist = TweedieDistribution(1.5)
t_glm1 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1,
                                    fit_intercept=True)
t_glm1.fit(X_train_t, y_train_t, sample_weight=w_train_t)


pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm1.intercept_], t_glm1.coef_))},
    index=["intercept"] + t_glm1.feature_names_,
).T

df_test["pp_t_glm1"] = t_glm1.predict(X_test_t)
df_train["pp_t_glm1"] = t_glm1.predict(X_train_t)

print(
    "training loss t_glm1:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm1"],
                             sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm1:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm1"],
                             sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * t_glm1.predict(X_test_t)),
    )
)
# %%
# TODO: Let's add splines for BonusMalus and Density and use a Pipeline.
# Steps:
# 1. Define a Pipeline which chains a StandardScaler and SplineTransformer.
#    Choose knots="quantile" for the SplineTransformer and make sure, we
#    are only including one intercept in the final GLM.
# 2. Put the transforms together into a ColumnTransformer.
# Here we use OneHotEncoder for the categoricals.
# 3. Chain the transforms together with the GLM in a Pipeline.

# Let's put together a pipeline
numeric_cols = ["BonusMalus", "Density"]
numeric_pipeline = Pipeline(
    steps=[
        ("scale", StandardScaler()),
        (
            "splines",
            SplineTransformer(
                degree=3,
                n_knots=5,
                knots="quantile",
                include_bias=False,
            ),
        ),
    ],
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        (
            "cat",
            OneHotEncoder(
                sparse_output=False,
                drop="first",
                handle_unknown="ignore",
            ),
            categoricals,
        ),
    ]
)
preprocessor.set_output(transform="pandas")
model_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        (
            "estimate",
            GeneralizedLinearRegressor(
                family=TweedieDist,
                fit_intercept=True,
                l1_ratio=1,
            ),
        ),
    ]
)

# let's have a look at the pipeline
model_pipeline

# let's check that the transforms worked
model_pipeline[:-1].fit_transform(df_train)

model_pipeline.fit(df_train, y_train_t, estimate__sample_weight=w_train_t)

pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([model_pipeline[-1].intercept_], model_pipeline[-1].coef_)
        )
    },
    index=["intercept"] + model_pipeline[-1].feature_names_,
).T

df_test["pp_t_glm2"] = model_pipeline.predict(df_test)
df_train["pp_t_glm2"] = model_pipeline.predict(df_train)

print(
    "training loss t_glm2:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm2"],
                             sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm2:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm2"],
                             sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_glm2"]),
    )
)

# %%
# TODO: Let's use a GBM instead as an estimator.
# Steps
# 1: Define the modelling pipeline. Tip: This can simply be a
# LGBMRegressor based on X_train_t from before.
# 2. Make sure we are choosing the correct objective for our estimator.
lgbm_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        (
            "estimate",
            LGBMRegressor(
                objective="tweedie",
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
model_pipeline = lgbm_pipeline
model_pipeline.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)
df_test["pp_t_lgbm"] = model_pipeline.predict(X_test_t)
df_train["pp_t_lgbm"] = model_pipeline.predict(X_train_t)
print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"],
                             sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"],
                             sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

# %%
# TODO: Let's tune the LGBM to reduce overfitting.
# Steps:
# 1. Define a `GridSearchCV` object with our lgbm pipeline/estimator.
# Tip: Parameters for a specific step of the pipeline
# can be passed by <step_name>__param.

# Note: Typically we tune many more parameters and larger grids,
# but to save compute time here, we focus on getting the learning rate
# and the number of estimators somewhat aligned -> tune learning_rate and n_estimators
param_grid = {
    "estimate__learning_rate": [0.1, 0.05, 0.01],
    "estimate__n_estimators": [300, 500, 1000],
}

cv = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring="neg_mean_poisson_deviance",
    cv=3,
    n_jobs=-1,
    verbose=1,
)
cv.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)

df_test["pp_t_lgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm"] = cv.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"],
                             sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm"]),
    )
)

print("Best params:", cv.best_params_)
print("Best estimator:", cv.best_estimator_)

# %%
# Let's compare the sorting of the pure premium predictions


# Source: https://scikit-learn.org/stable/auto_examples/linear_model
# /plot_tweedie_regression_insurance_claims.html
def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount


fig, ax = plt.subplots(figsize=(8, 8))

for label, y_pred in [
    ("LGBM", df_test["pp_t_lgbm"]),
    ("GLM Benchmark", df_test["pp_t_glm1"]),
    ("GLM Splines", df_test["pp_t_glm2"]),
]:
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], y_pred, df_test["Exposure"]
    )
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label += f" (Gini index: {gini: .3f})"
    ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# Oracle model: y_pred == y_test
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
gini = 1 - 2 * auc(ordered_samples, cum_claims)
label = f"Oracle (Gini index: {gini: .3f})"
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

# Random baseline
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz Curves",
    xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
    ylabel="Fraction of total claim amount",
)
ax.legend(loc="upper left")
plt.plot()

# %%
# This is the start of problem set 4 :)

# Exercise 1
# TODO: Create a plot of average claims (claimNB) per BonusMalus
print(df["BonusMalus"].unique())
print(df["ClaimAmount"].unique())
print(df["Exposure"].unique())
df[["BonusMalus","ClaimAmount","Exposure"]]

bonus_summary = (
    df.groupby("BonusMalus")
      .apply(lambda g: pd.Series({
          "total_claims": (g["ClaimAmount"] * g["Exposure"]).sum(),
          "total_exposure": g["Exposure"].sum()
      }))
)

bonus_summary["weighted_mean_claims"] = (
    bonus_summary["total_claims"] / bonus_summary["total_exposure"]
)

bonus_summary.head()


# %%
plt.figure(figsize=(10,5))
plt.plot(bonus_summary.index, bonus_summary["weighted_mean_claims"], marker="o", linestyle="-")
plt.xlabel("BonusMalus")
plt.ylabel("Weighted Mean Claim Amount")
plt.title("Weighted Average Claim Amount by BonusMalus")
plt.grid(True)
plt.show()

# %%
# TODO: Create a new model pipeline called constrained_lgbm
# TODO: Include the monotonic constrianted that [0,0,0,1,...] for whhen BonusMalus is 1
# The categorical and numerical columns we want our model to take as inputs

categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]
numeric_cols = ["BonusMalus", "Density"]
predictors = categoricals + numeric_cols
print(f"These are the input features: {predictors}")
print(f"If you're confused, these are the categorical columns we want to input in: {categoricals}")
print(f"and these are the numericals: {numeric_cols}")


# %%
# Addressing the preprocessing steps
numeric_pipeline = Pipeline(
    steps=[
        ("scale", StandardScaler()),
        (
            "splines",
            SplineTransformer(
                degree=3,
                n_knots=5,
                knots="quantile",
                include_bias=False,
            ),
        ),
    ],
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        (
            "cat",
            OneHotEncoder(
                sparse_output=False,
                drop="first",
                handle_unknown="ignore",
            ),
            categoricals,
        ),
    ]
)
preprocessor.set_output(transform="pandas")
preprocessor.fit(df_train)
feature_names = preprocessor.get_feature_names_out()
print(f"These are the feature names (the inputs for the model) in order: {feature_names}.")
print(f"Use the columns to understand which columns need monotonic constraints 1,0 or -1. The count of feature names is: {len(feature_names)}")

# %%
# Creating constrianted_lgbm that has the best params from GridSearchCV and the monotone constriants

monotonic_constraints = [
    # BonusMalus spline columns (6) → +1
    1, 1, 1, 1, 1, 1,

    # Density spline columns (6) → 0
    0, 0, 0, 0, 0, 0,

    # Categorical one-hot encoded columns (49) → 0
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0
]

constrained_lgbm = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        (
            "estimate",
            LGBMRegressor(
                learning_rate=0.01,
                n_estimators=300,
                objective='tweedie',
                tweedie_variance_power=1.5,
                colsample_bytree=0.8,
                subsample=0.8,
                min_data_in_leaf=200,
                monotone_constraints=monotonic_constraints
            ),
        ),
    ]
)


# %%
# Fitting the pipeline and printing training and testing unit deviance = scalar loss = model performance metric
constrained_lgbm.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)
df_test["pp_t_constrained_lgbm"] = constrained_lgbm.predict(X_test_t)
df_train["pp_t_constrained_lgbm"] = constrained_lgbm.predict(X_train_t)
print(
    "training loss t_constrained_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_constrained_lgbm"],
                             sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_constrained_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_constrained_lgbm"],
                             sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)
# %%
# Displaying the train and test dataset for my pleasure
df_train.head()
# %%
# Displaying the train and test dataset for my pleasure
df_test.head()
print(f"The length of constraints to training data shape: {len(monotonic_constraints), X_train_t.shape[1]}")
# %% Exercise 2 from Problem Set 4
# Refitting the LGBM with constrained and best params with our eval_set and eval_metric
lgbm_eval = LGBMRegressor(
        learning_rate=0.01,
        n_estimators=300,
        objective='tweedie',
        tweedie_variance_power=1.5,
        colsample_bytree=0.8,
        subsample=0.8,
        min_data_in_leaf=200
        )

# %%
# Fitting the lgbm_eval model removing monotone constriants because it was breaking the kernel
evals_result = {}
lgbm_eval.fit(
    X_train_t,
    y_train_t,
    sample_weight=w_train_t,
    eval_set=[(X_train_t, y_train_t), (X_test_t, y_test_t)],
    eval_metric="mean_poisson_deviance",
    callbacks=[lgb.record_evaluation(evals_result)]
)

df_test["pp_t_lgbm_eval"] = lgbm_eval.predict(X_test_t)
df_train["pp_t_lgbm_eval"] = lgbm_eval.predict(X_train_t)
# %%
# Extract the underlying LightGBM booster
lgb.plot_metric(evals_result)
print(f"From the plot, we understand: ")
print("- We have two lines, one for training and one for testing dataset")
print("- The numbers on the Y axis come from the evaluation metric which is the unit deviance = scalar loss = poisson deviance. Even through our tweedie variance is 1.5 so inbetween Gamma and Poisson, we still use poisson deviance because that’s the standard that LGBM uses.")
print("- X axis is the number of iterations, in the model we stated 300 estimations. This means every iteration is a tree, so x is the number of trees trained.")
print("- Early on, the model performance is 80 for testing and 47 ish for training. As we add more iterations, the model performance falls to less than 45. The test performance goes to 75. It means there are no sudden changes so the model does not overfit or underfit. Although the 47 is quite low, could be due to the monotonic constraint issue. The model is optimally fitted.")
# %%
# Using the functions from _evaluate_predictions to compare the models of constrained_lgbm vs lgbm_eval
results_constrained = evaluate_predictions(
    y_true=df_test["PurePremium"],
    y_pred=df_test["pp_t_constrained_lgbm"],
    sample_weight=df_test["Exposure"]
)

results_unconstrained = evaluate_predictions(
    y_true=df_test["PurePremium"],
    y_pred=df_test["pp_t_lgbm_eval"],
    sample_weight=df_test["Exposure"]
)

# %%
print(f"The results for LGBM constrained: \n{results_constrained}")
print(f"The results for LGBM unconstrained: \n{results_unconstrained}")

# %%
# Using the explanier object to undersand the marginal effects of specific features for the constrained LGBM - Exercise 4 for PS4
explainer_constrained = dx.Explainer(
    model = constrained_lgbm,
    data = df_test[predictors],
    y = df_test["PurePremium"],
    label = "Constrained LGBM"
)

# %%
# Creating the plot
pd_constrained = explainer_constrained.model_profile(
    variables=predictors
)
pd_constrained.plot()
# %%
