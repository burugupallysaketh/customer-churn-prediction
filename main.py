
# ==========================================
# CUSTOMER CHURN PREDICTION PROJECT
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("=" * 50)
print("CUSTOMER CHURN PREDICTION PROJECT")
print("=" * 50)

# 1. Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

print("\nTotal customers:", len(df))
print("Total columns:", len(df.columns))

# 2. Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

print("\nMissing values cleaned.")

# 3. Check churn distribution
print("\nCustomer churn count:")
print(df["Churn"].value_counts())

# 4. Separate features and target
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

# Preprocess numerical and categorical columns
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining customers:", len(X_train))
print("Testing customers:", len(X_test))

# 6. Build complete preprocessing and model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

# 7. Train the model
model.fit(X_train, y_train)

print("\nMachine Learning model trained successfully!")

# 8. Evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Stays", "Leaves"],
    zero_division=0
))

# 9. Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted Stays", "Predicted Leaves"],
    yticklabels=["Actual Stays", "Actual Leaves"]
)

plt.title("Customer Churn Prediction - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# 10. Feature importance
# Get transformed feature names from the fitted pipeline
fitted_preprocessor = model.named_steps["preprocessor"]
fitted_classifier = model.named_steps["classifier"]

feature_names = fitted_preprocessor.get_feature_names_out()

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": fitted_classifier.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(importance.head(10).to_string(index=False))

importance.to_csv("feature_importance.csv", index=False)

# 11. Save test predictions
results = X_test.copy()
results["Actual_Churn"] = y_test.values
results["Predicted_Churn"] = y_pred

results.to_csv("churn_predictions.csv", index=False)

# 12. Save the complete pipeline
joblib.dump(model, "churn_model.pkl")

print("\nFiles saved:")
print("1. confusion_matrix.png")
print("2. churn_predictions.csv")
print("3. feature_importance.csv")
print("4. churn_model.pkl")

print("\nComplete preprocessing pipeline and model saved!")
print("\nCUSTOMER CHURN MACHINE LEARNING COMPLETED!")
