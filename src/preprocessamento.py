import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import logging
import argparse
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_preprocess_data(file_path: str):
    df = pd.read_csv(file_path, low_memory=False)
    logging.info(f"Loaded data shape: {df.shape}")

    numeric_columns = []
    categorical_columns = []
    date_columns = ['primeiraCompra', 'dataAprovadoEmComite', 'dataAprovadoNivelAnalista', 'periodoBalanco']

    for col in df.columns:
        if col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        elif df[col].dtype == 'object' or df[col].dtype == 'bool':
            categorical_columns.append(col)
            df[col] = df[col].astype(str)  # Convert bool to str
        elif df[col].dtype in ['int64', 'float64']:
            numeric_columns.append(col)

    for col in date_columns:
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        numeric_columns.extend([f'{col}_year', f'{col}_month'])

    logging.info(f"Numeric columns: {numeric_columns}")
    logging.info(f"Categorical columns: {categorical_columns}")
    logging.info(f"Date columns: {date_columns}")

    return df, numeric_columns, categorical_columns


def create_preprocessor(numeric_columns, categorical_columns):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    return preprocessor


def main(input_path: str, output_path: str):
    try:
        df, numeric_columns, categorical_columns = load_and_preprocess_data(input_path)

        preprocessor = create_preprocessor(numeric_columns, categorical_columns)

        X = preprocessor.fit_transform(df)

        cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
            categorical_columns).tolist()
        feature_names = numeric_columns + cat_feature_names

        X_df = pd.DataFrame(X, columns=feature_names)

        X_df.to_csv(output_path, index=False)
        joblib.dump(preprocessor, os.path.splitext(output_path)[0] + '_preprocessor.joblib')

        logging.info(f"Preprocessed data shape: {X_df.shape}")
        logging.info(f"Preprocessed data saved to {output_path}")
        logging.info(f"Preprocessor saved to {os.path.splitext(output_path)[0] + '_preprocessor.joblib'}")

    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {str(e)}")
        logging.error("Error details: ", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing Script")
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--output", required=True, help="Path to save the preprocessed data")
    args = parser.parse_args()

    main(args.input, args.output)