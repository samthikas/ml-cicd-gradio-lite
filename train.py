import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load dataset
df = pd.read_csv("data/drug200.csv")
df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Separate features and target
X = df.drop(columns=["Drug"])
y = df["Drug"]

categorical_cols = ["Sex", "BP", "Cholesterol"]
numeric_cols = ["Age", "Na_to_K"]

# Preprocess and train model
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("numeric", StandardScaler(), numeric_cols),
    ]
)

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="weighted")

# Save trained model and evaluation results
joblib.dump(pipeline, "model/drug_model.joblib")

with open("results/metrics.txt", "w", encoding="utf-8") as file:
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Weighted F1 Score: {f1:.4f}\n")

ConfusionMatrixDisplay.from_predictions(y_test, predictions, xticks_rotation="vertical")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

print(f"Accuracy: {accuracy:.4f}")
print(f"Weighted F1 Score: {f1:.4f}")
print("Saved model to model/drug_model.joblib")
print("Saved metrics to results/metrics.txt")
print("Saved confusion matrix to results/confusion_matrix.png")

# Export model to ONNX for browser-based inference
try:
    initial_types = [
        ("Sex", StringTensorType([None, 1])),
        ("BP", StringTensorType([None, 1])),
        ("Cholesterol", StringTensorType([None, 1])),
        ("Age", FloatTensorType([None, 1])),
        ("Na_to_K", FloatTensorType([None, 1])),
    ]

    
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_types,
        target_opset=17,
        options={id(pipeline.steps[-1][1]): {"zipmap": False}},
    )


    with open("model/drug_model.onnx", "wb") as file:
        file.write(onnx_model.SerializeToString())

    print("Saved ONNX model to model/drug_model.onnx")

except Exception as error:
    print("ONNX export failed:", error)
    print("The trained joblib model and evaluation results are still saved.")
