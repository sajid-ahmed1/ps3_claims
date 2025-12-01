# %%
import dalex as dx
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from ps3.data import create_sample_split, load_transform
from ps3.evaluation import evaluate_predictions, lorenz_curve

# %%
# load data
df = load_transform()

# %%
# Train benchmark tweedie model
weight = df["Exposure"].values
df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
y = df["PurePremium"]

df = create_sample_split(df, id_column="IDpol")
train = np.where(df["sample"] == "train")
test = np.where(df["sample"] == "test")
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()

categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]

predictors = categoricals + ["BonusMalus", "Density"]
glm_categorizer = Categorizer(columns=categoricals)

X_train_t = glm_categorizer.fit_transform(df[predictors].iloc[train])
X_test_t = glm_categorizer.transform(df[predictors].iloc[test])
y_train_t, y_test_t = y.iloc[train], y.iloc[test]
w_train_t, w_test_t = weight[train], weight[test]

TweedieDist = TweedieDistribution(1.5)
t_glm1 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True)
t_glm1.fit(X_train_t, y_train_t, sample_weight=w_train_t)


pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm1.intercept_], t_glm1.coef_))},
    index=["intercept"] + t_glm1.feature_names_,
).T

df_test["pp_t_glm1"] = t_glm1.predict(X_test_t)
df_train["pp_t_glm1"] = t_glm1.predict(X_train_t)

pd.set_option("display.float_format", lambda x: "%.3f" % x)
evaluate_predictions(
    df_test,
    outcome_column="PurePremium",
    exposure_column="Exposure",
    preds_column="pp_t_glm1",
)
# %%
# Let's add splines for BonusMalus and Density and use a pipeline

# Let's put together pipeline
numeric_cols = ["BonusMalus", "Density"]
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("spline", SplineTransformer(include_bias=False, knots="quantile")),
                ]
            ),
            numeric_cols,
        ),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals),
    ]
)
preprocessor.set_output(transform="pandas")
model_pipeline = Pipeline(
    [
        ("preprocess", preprocessor),
        (
            "estimate",
            GeneralizedLinearRegressor(
                family=TweedieDist, l1_ratio=1, fit_intercept=True
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

evaluate_predictions(
    df_test,
    outcome_column="PurePremium",
    exposure_column="Exposure",
    preds_column="pp_t_glm2",
)

# %%
# Let's use a GBM instead as an estimator
model_pipeline = Pipeline([("estimate", LGBMRegressor(objective="tweedie"))])

model_pipeline.fit(
    X_train_t,
    y_train_t,
    estimate__sample_weight=w_train_t,
    estimate__eval_set=[(X_test_t, y_test_t), (X_train_t, y_train_t)],
)
df_test["pp_t_lgbm"] = model_pipeline.predict(X_test_t)
df_train["pp_t_lgbm"] = model_pipeline.predict(X_train_t)
evaluate_predictions(
    df_test,
    outcome_column="PurePremium",
    exposure_column="Exposure",
    preds_column="pp_t_lgbm",
)

# %%
model_pipeline = Pipeline(
    [
        (
            "estimate",
            LGBMRegressor(
                objective="tweedie",
                n_estimators=1000,
                learning_rate=0.1,
                num_leaves=6,
                early_stopping_rounds=25,
            ),
        )
    ]
)
model_pipeline.fit(
    X_train_t,
    y_train_t,
    estimate__sample_weight=w_train_t,
    estimate__eval_set=[(X_test_t, y_test_t), (X_train_t, y_train_t)],
)
lgb.plot_metric(model_pipeline[0])

df_test["pp_t_lgbm"] = model_pipeline.predict(X_test_t)
df_train["pp_t_lgbm"] = model_pipeline.predict(X_train_t)
evaluate_predictions(
    df_test,
    outcome_column="PurePremium",
    exposure_column="Exposure",
    preds_column="pp_t_lgbm",
)

# %%
# Let's tune the pipeline to reduce overfitting

# Note: Typically we tune many more parameters and larger grids,
# but to save compute time here, we focus on getting the learning rate
# and the number of estimators somewhat aligned
cv = GridSearchCV(
    model_pipeline,
    {
        "estimate__learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05, 0.1],
        "estimate__n_estimators": [50, 100, 150, 200],
    },
    verbose=2,
)
cv.fit(
    X_train_t,
    y_train_t,
    estimate__sample_weight=w_train_t,
    estimate__eval_set=[(X_test_t, y_test_t), (X_train_t, y_train_t)],
)

lgbm_unconstrained = cv.best_estimator_
df_test["pp_t_lgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm"] = cv.best_estimator_.predict(X_train_t)

evaluate_predictions(
    df_test,
    outcome_column="PurePremium",
    exposure_column="Exposure",
    preds_column="pp_t_lgbm",
)

# %%

df_plot = (
    df.groupby("BonusMalus")
    .apply(lambda x: np.average(x["PurePremium"], weights=x["Exposure"]))
    .reset_index(name="PurePremium")
)

sns.scatterplot(df_plot, x="BonusMalus", y="PurePremium")

# %%

# Constrained LGBM

lgbm_constrained = Pipeline(
    [
        (
            "estimate",
            LGBMRegressor(
                objective="tweedie", monotone_constraints=[0, 0, 0, 0, 0, 0, 0, 1, 0]
            ),
        )
    ]
)
cv = GridSearchCV(
    lgbm_constrained,
    {
        "estimate__learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05, 0.1],
        "estimate__n_estimators": [5000],
    },
    verbose=2,
)
cv.fit(
    X_train_t,
    y_train_t,
    estimate__sample_weight=w_train_t,
    estimate__eval_set=[(X_test_t, y_test_t), (X_train_t, y_train_t)],
)


df_test["pp_t_lgbm_constrained"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm_constrained"] = cv.best_estimator_.predict(X_train_t)

# %%
# Plot learning curve
lgbm_constrained.fit(
    X_train_t,
    y_train_t,
    estimate__eval_set=[(X_test_t, y_test_t), (X_train_t, y_train_t)],
)

lgb.plot_metric(lgbm_constrained[0])

# %%
evaluate_predictions(
    df_test,
    outcome_column="PurePremium",
    exposure_column="Exposure",
    preds_column="pp_t_lgbm_constrained",
)

# %%
# Plot PDPs

lgbm_constrained_exp = dx.Explainer(
    lgbm_constrained, X_test_t, y_test_t, label="Constrained LGBM"
)
pdp_constrained = lgbm_constrained_exp.model_profile()

lgbm_unconstrained_exp = dx.Explainer(
    lgbm_unconstrained, X_test_t, y_test_t, label="Unconstrained LGBM"
)
pdp_unconstrained = lgbm_unconstrained_exp.model_profile()

pdp_constrained.plot(pdp_unconstrained)

# %%
shap = lgbm_constrained_exp.predict_parts(X_test_t.head(1), type="shap")

shap.plot()
# %%
# Let's compare the sorting of the pure premium predictions


fig, ax = plt.subplots(figsize=(8, 8))

for label, y_pred in [
    ("LGBM", df_test["pp_t_lgbm"]),
    ("LGBM_constrained", df_test["pp_t_lgbm_constrained"]),
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
