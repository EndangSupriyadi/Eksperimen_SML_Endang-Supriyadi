import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from joblib import dump


def automate_preprocessing(
        data,                 
        target_column,        
        save_pipeline_path,   
        save_train_path,      
        save_test_path        
    ):

    print("=== START AUTOMATED PREPROCESSING ===")

    # ---------------------------------------------------
    # 1. BASIC CLEANING (MANUAL CLEANING)
    # ---------------------------------------------------

    data = data.replace(r'^\s*$', np.nan, regex=True)

    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    data = data.dropna(subset=["TotalCharges"])

    cols_internet = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]

    for col in cols_internet:
        if col in data.columns:
            data[col] = data[col].replace("No internet service", "No")

    if "MultipleLines" in data.columns:
        data["MultipleLines"] = data["MultipleLines"].replace("No phone service", "No")

    if "customerID" in data.columns:
        data = data.drop("customerID", axis=1)

    print("✔ Basic cleaning selesai")

    # ---------------------------------------------------
    # 2. IDENTIFIKASI FITUR NUMERIK & KATEGORIS
    # ---------------------------------------------------

    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()

    if target_column in numeric_features:
        numeric_features.remove(target_column)

    if target_column in categorical_features:
        categorical_features.remove(target_column)

    # ---------------------------------------------------
    # 3. PIPELINE NUMERIK & KATEGORI
    # ---------------------------------------------------

    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='passthrough'
    )

    # ---------------------------------------------------
    # 4. SPLIT DATA
    # ---------------------------------------------------

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("✔ Train-Test split selesai")
    print("Train size:", X_train.shape)
    print("Test size :", X_test.shape)

    # ---------------------------------------------------
    # 5. FIT ONLY ON TRAIN → TRANSFORM
    # ---------------------------------------------------

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("✔ Transformasi dengan pipeline selesai")

    # ---------------------------------------------------
    # 6. SMOTE-TOMEK
    # ---------------------------------------------------

    smote_tomek = SMOTETomek(random_state=42)
    X_train_balanced, y_train_balanced = smote_tomek.fit_resample(
        X_train_processed, y_train
    )

    print("✔ SMOTE-Tomek selesai")
    print("Jumlah data sebelum :", y_train.value_counts().to_dict())
    print("Jumlah data sesudah :", y_train_balanced.value_counts().to_dict())

    # ---------------------------------------------------
    # 7. SIMPAN PIPELINE & DATA HASIL
    # ---------------------------------------------------

    dump(preprocessor, save_pipeline_path)
    print(f"✔ Pipeline disimpan ke: {save_pipeline_path}")

    pd.DataFrame(X_train_balanced).assign(Churn=y_train_balanced)\
        .to_csv(save_train_path, index=False)
    print(f"✔ Train CSV disimpan ke: {save_train_path}")

    pd.DataFrame(X_test_processed).assign(Churn=y_test)\
        .to_csv(save_test_path, index=False)
    print(f"✔ Test CSV disimpan ke: {save_test_path}")

    print("=== AUTOMATED PREPROCESSING DONE ===")
    return X_train_balanced, X_test_processed, y_train_balanced, y_test


# ---------------------------------------------------
# 8. PEMANGGILAN FUNGSI 
# ---------------------------------------------------
if __name__ == "__main__":

    # Path ke dataset mentah
    raw_data_path = "./dataset/Telco-Customer-Churn.csv"
    data = pd.read_csv(raw_data_path)

    # Kolom target
    target_column = "Churn"

    # >>> WAJIB DIISI MANUAL <<<
    save_pipeline_path = "./preprocessing/preprocessing_pipeline.joblib"
    save_train_path = "./preprocessing/train_preprocessed.csv"
    save_test_path = "./preprocessing/test_preprocessed.csv"

    automate_preprocessing(
        data=data,
        target_column=target_column,
        save_pipeline_path=save_pipeline_path,
        save_train_path=save_train_path,
        save_test_path=save_test_path
    )
