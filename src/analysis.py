import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "doctor_visits.csv")
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(DATA)

# ---------------------------
# 1. Data quality
# ---------------------------
quality = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing": df.isna().sum(),
    "unique": df.nunique()
})
quality.to_csv(os.path.join(OUT, "data_quality.csv"))

# ---------------------------
# 2. Basic statistics
# ---------------------------
df.describe(include="all").T.to_csv(os.path.join(OUT, "descriptive_statistics.csv"))

# ---------------------------
# 3. EDA charts
# ---------------------------
plt.figure(figsize=(8,5))
vc = df["visits"].value_counts().sort_index()
plt.bar(vc.index.astype(str), vc.values)
plt.title("Distribution of Doctor Visits")
plt.xlabel("Number of Doctor Visits")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_visits_distribution.png"), dpi=160)
plt.close()

plt.figure(figsize=(8,5))
plt.hist(df["age"], bins=20, edgecolor="black")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_age_distribution.png"), dpi=160)
plt.close()

plt.figure(figsize=(8,5))
groups = [df.loc[df.gender == g, "visits"].values for g in sorted(df.gender.unique())]
plt.boxplot(groups, labels=sorted(df.gender.unique()))
plt.title("Doctor Visits by Gender")
plt.xlabel("Gender")
plt.ylabel("Doctor Visits")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "03_visits_by_gender.png"), dpi=160)
plt.close()

plt.figure(figsize=(8,5))
plt.scatter(df["reduced"], df["visits"], alpha=0.35)
plt.title("Reduced Activity Days vs Doctor Visits")
plt.xlabel("Reduced Activity Days")
plt.ylabel("Doctor Visits")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_reduced_vs_visits.png"), dpi=160)
plt.close()

plt.figure(figsize=(8,5))
illness_groups = [df.loc[df.illness == i, "visits"].values for i in sorted(df.illness.unique())]
plt.boxplot(illness_groups, labels=sorted(df.illness.unique()))
plt.title("Doctor Visits by Number of Illnesses")
plt.xlabel("Number of Illnesses")
plt.ylabel("Doctor Visits")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "05_illness_vs_visits.png"), dpi=160)
plt.close()

plt.figure(figsize=(9,7))
corr = df.select_dtypes(include=np.number).drop(columns=["Unnamed: 0"]).corr()
plt.imshow(corr, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
plt.title("Numerical Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "06_correlation_heatmap.png"), dpi=160)
plt.close()

# ---------------------------
# 4. Group analysis
# ---------------------------
group_results = {
    "gender_mean_visits": df.groupby("gender")["visits"].mean().sort_values(ascending=False),
    "nchronic_mean_visits": df.groupby("nchronic")["visits"].mean().sort_values(ascending=False),
    "lchronic_mean_visits": df.groupby("lchronic")["visits"].mean().sort_values(ascending=False),
    "private_mean_visits": df.groupby("private")["visits"].mean().sort_values(ascending=False),
    "freepoor_mean_visits": df.groupby("freepoor")["visits"].mean().sort_values(ascending=False),
    "freerepat_mean_visits": df.groupby("freerepat")["visits"].mean().sort_values(ascending=False),
}
with open(os.path.join(OUT, "group_analysis.txt"), "w") as f:
    for name, result in group_results.items():
        f.write("\n" + name + "\n")
        f.write(result.to_string() + "\n")

# ---------------------------
# 5. Predictive modeling
# ---------------------------
X = df.drop(columns=["visits", "Unnamed: 0"])
y = df["visits"]

categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric = X.select_dtypes(include=np.number).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )
}

rows = []
for name, estimator in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    rows.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

results = pd.DataFrame(rows)
results.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)

# Save predictions from Random Forest
rf_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", models["Random Forest"])
])
rf_pipe.fit(X_train, y_train)
pred = rf_pipe.predict(X_test)
predictions = X_test.copy()
predictions["actual_visits"] = y_test.values
predictions["predicted_visits"] = pred
predictions.to_csv(os.path.join(OUT, "test_predictions.csv"), index=False)

print("Analysis complete.")
print(results.to_string(index=False))
