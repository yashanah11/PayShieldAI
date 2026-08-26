import joblib
import os

path = "models/xgboost_detector.joblib"

print("=== STAGE 2D: MODEL PROVENANCE AUDIT ===")

print("Model path :", os.path.abspath(path))
print("Exists     :", os.path.exists(path))

if not os.path.exists(path):
    raise FileNotFoundError(path)

model = joblib.load(path)

print("\n=== MODEL TYPE ===")
print(type(model))

print("\n=== MODEL PARAMETERS ===")
try:
    print(model.get_params())
except Exception as e:
    print("Could not read parameters:", e)

print("\n=== MODEL FEATURES ===")
try:
    print(model.feature_names_in_)
except Exception as e:
    print("No feature_names_in_:", e)

print("\n=== MODEL CLASSES ===")
try:
    print(model.classes_)
except Exception as e:
    print("Could not read classes:", e)

print("\n=== MODEL ESTIMATORS ===")
try:
    print("n_estimators:", model.n_estimators)
except Exception:
    pass

print("\n=== FILE INFO ===")
print("Size:", os.path.getsize(path), "bytes")
print("Modified:", os.path.getmtime(path))

print("\n=== PROVENANCE AUDIT COMPLETE ===")
