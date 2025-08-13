import pandas as pd

import joblib



def classify_berth(input_csv, output_csv, model_path='model.joblib', features_path='feature_columns.joblib'):

    # Load the saved model

    model = joblib.load(model_path)

    print(f"Loaded model from {model_path}")



    # Load saved feature columns list that was used during training

    feature_columns = joblib.load(features_path)

    print(f"Loaded feature columns from {features_path}, total {len(feature_columns)} features")



    # Load input data

    df = pd.read_csv(input_csv)

    print(f"Loaded input data with {len(df)} rows from {input_csv}")



    # Drop any columns that are not features (e.g., 'File' or target columns) if present

    # Because training data does not have 'File' column

    if 'File' in df.columns:

        df = df.drop(columns=['File'])



    # Columns expected to be categorical for one-hot encoding

    categorical_cols = ['Cargo Type', 'Flag', 'Single_Hull', 'Double_Sides', 'Double_Bottoms']



    # Ensure hull boolean columns are string, as in training (usually 'True'/'False')

    for col in ['Single_Hull', 'Double_Sides', 'Double_Bottoms']:

        if col in df.columns:

            df[col] = df[col].astype(str)

        else:

            raise ValueError(f"Expected column '{col}' not found in input CSV")



    # One-hot encode categorical columns

    df_encoded = pd.get_dummies(df, columns=categorical_cols)



    # Add missing columns (features seen during training but missing in this input data) with zeros

    missing_cols = [col for col in feature_columns if col not in df_encoded.columns]

    for col in missing_cols:

        df_encoded[col] = 0



    # Remove columns present in input but not seen during training

    extra_cols = [col for col in df_encoded.columns if col not in feature_columns]

    if extra_cols:

        df_encoded = df_encoded.drop(columns=extra_cols)



    # Reorder columns to match training order exactly

    df_encoded = df_encoded[feature_columns]



    # Predict using the loaded model

    predictions = model.predict(df_encoded)



    # Add predictions to original dataframe (before dropping columns)

    df['Predicted_Assigned_Berth_Grouped'] = predictions



    # Save the dataframe with predictions to CSV

    df.to_csv(output_csv, index=False)

    print(f"Predictions saved to {output_csv}")





if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Classify Berth Group using Random Forest model")

    parser.add_argument("input_csv", help="CSV file path with features (relevant headers)")

    parser.add_argument("output_csv", help="Output CSV file path to save predictions")

    parser.add_argument("--model_path", default="model.joblib", help="Path to the saved RF model file")

    parser.add_argument("--features_path", default="feature_columns.joblib", help="Path to the saved list of feature columns")



    args = parser.parse_args()



    classify_berth(args.input_csv, args.output_csv, args.model_path, args.features_path)