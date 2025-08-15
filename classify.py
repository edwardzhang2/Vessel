import pandas as pd
import joblib
import sys

def classify_berth(input_csv, output_csv, model_path='model.joblib', features_path='feature_columns.joblib'):
    # Load input data first so we can short-circuit early
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print(f"[classify] Loaded input data with {len(df)} rows from {input_csv}")

    if df.empty:
        print("[classify] Input is empty. Writing empty output with expected columns and exiting cleanly.")
        # Write empty output with a minimal schema (or reuse original headers if you prefer)
        df_out = df.copy()
        if "Predicted_Assigned_Berth_Grouped" not in df_out.columns:
            df_out["Predicted_Assigned_Berth_Grouped"] = []
        df_out.to_csv(output_csv, index=False)
        print(f"[classify] Saved empty output to {output_csv}")
        return

    # Load model + feature list
    model = joblib.load(model_path)
    print(f"[classify] Loaded model from {model_path}")
    feature_columns = joblib.load(features_path)
    print(f"[classify] Loaded feature columns from {features_path}, total {len(feature_columns)} features")

    # Drop non-feature columns if present
    if 'File' in df.columns:
        df = df.drop(columns=['File'])

    # Categorical columns expected in training
    categorical_cols = ['Cargo Type', 'Flag', 'Single_Hull', 'Double_Sides', 'Double_Bottoms']

    # Ensure hull booleans are string-typed (as in training)
    for col in ['Single_Hull', 'Double_Sides', 'Double_Bottoms']:
        if col in df.columns:
            df[col] = df[col].astype(str)
        else:
            # If missing, add as False -> "False"
            df[col] = "False"

    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    # Add missing training columns with zeros
    missing_cols = [col for col in feature_columns if col not in df_encoded.columns]
    for col in missing_cols:
        df_encoded[col] = 0

    # Drop extra columns not seen in training
    extra_cols = [col for col in df_encoded.columns if col not in feature_columns]
    if extra_cols:
        df_encoded = df_encoded.drop(columns=extra_cols)

    # Reorder exactly as training
    df_encoded = df_encoded[feature_columns]

    if df_encoded.shape[0] == 0:
        print("[classify] Encoded matrix has 0 rows. Writing empty output.")
        df_out = df.copy()
        df_out["Predicted_Assigned_Berth_Grouped"] = []
        df_out.to_csv(output_csv, index=False)
        print(f"[classify] Saved empty output to {output_csv}")
        return

    # Predict
    predictions = model.predict(df_encoded)

    # Save
    df["Predicted_Assigned_Berth_Grouped"] = predictions
    df.to_csv(output_csv, index=False)
    print(f"[classify] Predictions saved to {output_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Classify Berth Group using Random Forest model")
    parser.add_argument("input_csv", help="CSV file path with features (relevant headers)")
    parser.add_argument("output_csv", help="Output CSV file path to save predictions")
    parser.add_argument("--model_path", default="model.joblib", help="Path to the saved RF model file")
    parser.add_argument("--features_path", default="feature_columns.joblib", help="Path to the saved list of feature columns")
    args = parser.parse_args()

    classify_berth(args.input_csv, args.output_csv, args.model_path, args.features_path)
