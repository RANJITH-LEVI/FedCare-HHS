"""
FedCare-HHS Preprocessing Pipeline
Handles data ingestion, missing-value imputation, feature scaling,
and SMOTE balancing for the UCI Heart Disease datasets.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]
TARGET_NAME = 'target'

# Clinical descriptions and reference ranges
FEATURE_METADATA = {
    'age': {'label': 'Age', 'unit': 'years', 'type': 'numeric', 'min': 20, 'max': 90, 'default': 54.0},
    'sex': {'label': 'Sex', 'unit': 'binary', 'type': 'categorical', 'options': {0: 'Female', 1: 'Male'}, 'default': 1},
    'cp': {'label': 'Chest Pain Type', 'unit': 'grade', 'type': 'categorical',
           'options': {1: 'Typical Angina', 2: 'Atypical Angina', 3: 'Non-Anginal Pain', 4: 'Asymptomatic'}, 'default': 3},
    'trestbps': {'label': 'Resting Blood Pressure', 'unit': 'mm Hg', 'type': 'numeric', 'min': 80, 'max': 220, 'default': 130.0},
    'chol': {'label': 'Serum Cholesterol', 'unit': 'mg/dl', 'type': 'numeric', 'min': 100, 'max': 600, 'default': 240.0},
    'fbs': {'label': 'Fasting Blood Sugar > 120 mg/dl', 'unit': 'binary', 'type': 'categorical', 'options': {0: 'False (<= 120)', 1: 'True (> 120)'}, 'default': 0},
    'restecg': {'label': 'Resting ECG Results', 'unit': 'category', 'type': 'categorical',
                'options': {0: 'Normal', 1: 'ST-T Wave Abnormality', 2: 'Left Ventricular Hypertrophy'}, 'default': 0},
    'thalach': {'label': 'Maximum Heart Rate Achieved', 'unit': 'bpm', 'type': 'numeric', 'min': 60, 'max': 220, 'default': 150.0},
    'exang': {'label': 'Exercise-Induced Angina', 'unit': 'binary', 'type': 'categorical', 'options': {0: 'No', 1: 'Yes'}, 'default': 0},
    'oldpeak': {'label': 'ST Depression (Exercise vs Rest)', 'unit': 'mm', 'type': 'numeric', 'min': 0.0, 'max': 7.0, 'default': 1.0},
    'slope': {'label': 'Slope of Peak Exercise ST', 'unit': 'category', 'type': 'categorical',
              'options': {1: 'Upsloping', 2: 'Flat', 3: 'Downsloping'}, 'default': 1},
    'ca': {'label': 'Number of Major Vessels (0-3)', 'unit': 'count', 'type': 'numeric', 'min': 0, 'max': 3, 'default': 0},
    'thal': {'label': 'Thalassemia', 'unit': 'category', 'type': 'categorical',
             'options': {3: 'Normal (3)', 6: 'Fixed Defect (6)', 7: 'Reversible Defect (7)'}, 'default': 3}
}

HOSPITALS_INFO = {
    'cleveland': {
        'id': 'HOSP-01',
        'name': 'Cleveland Clinic Foundation',
        'location': 'Cleveland, OH, USA',
        'raw_file': 'data/raw/cleveland.csv',
        'description': 'Primary gold-standard cardiology center cohort'
    },
    'hungarian': {
        'id': 'HOSP-02',
        'name': 'Hungarian Institute of Cardiology',
        'location': 'Budapest, Hungary',
        'raw_file': 'data/raw/hungarian.csv',
        'description': 'Central European cardiac inpatient cohort'
    },
    'switzerland': {
        'id': 'HOSP-03',
        'name': 'University Hospital Zurich',
        'location': 'Zurich, Switzerland',
        'raw_file': 'data/raw/switzerland.csv',
        'description': 'Alpine regional referral center cohort'
    },
    'va': {
        'id': 'HOSP-04',
        'name': 'V.A. Medical Center',
        'location': 'Long Beach, CA, USA',
        'raw_file': 'data/raw/va.csv',
        'description': 'Veterans health administration outpatient cohort'
    }
}


class ClinicalDataPipeline:
    """
    Reusable data preprocessing pipeline for heart disease data.
    Provides strict fit-on-train and leakage-free transform.
    """
    def __init__(self, selected_features: Optional[List[str]] = None, use_smote: bool = True):
        self.selected_features = selected_features or FEATURE_NAMES.copy()
        self.use_smote = use_smote
        self.num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        self.cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        clean_df = df.copy()
        # In Hungarian, Swiss, and VA cohorts, chol=0 was entered for missing test
        if 'chol' in clean_df.columns:
            clean_df['chol'] = clean_df['chol'].replace(0, np.nan)
        # Ensure all expected columns exist
        for col in FEATURE_NAMES:
            if col not in clean_df.columns:
                clean_df[col] = np.nan
        return clean_df

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'ClinicalDataPipeline':
        X_clean = self._clean_dataframe(X)
        
        # Ensure default physiological medians exist if a site has 100% missing values
        default_fallbacks = {'age': 54.0, 'trestbps': 130.0, 'chol': 240.0, 'thalach': 150.0, 'oldpeak': 1.0}
        for col, val in default_fallbacks.items():
            if col in X_clean.columns and X_clean[col].isna().all():
                X_clean[col] = val

        active_num = [c for c in self.num_cols if c in X_clean.columns]
        active_cat = [c for c in self.cat_cols if c in X_clean.columns]
        
        self.num_imputer.fit(X_clean[active_num])
        self.cat_imputer.fit(X_clean[active_cat])
        
        # Intermediate imputed data
        X_num = pd.DataFrame(self.num_imputer.transform(X_clean[active_num]), columns=active_num, index=X_clean.index)
        X_cat = pd.DataFrame(self.cat_imputer.transform(X_clean[active_cat]), columns=active_cat, index=X_clean.index)
        X_combined = pd.concat([X_num, X_cat], axis=1)[FEATURE_NAMES]
        
        # Subset to selected features
        X_sub = X_combined[self.selected_features]
        self.scaler.fit(X_sub)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transforming data.")
        X_clean = self._clean_dataframe(X)
        
        active_num = [c for c in self.num_cols if c in X_clean.columns]
        active_cat = [c for c in self.cat_cols if c in X_clean.columns]
        
        X_num = pd.DataFrame(self.num_imputer.transform(X_clean[active_num]), columns=active_num, index=X_clean.index)
        X_cat = pd.DataFrame(self.cat_imputer.transform(X_clean[active_cat]), columns=active_cat, index=X_clean.index)
        X_combined = pd.concat([X_num, X_cat], axis=1)[FEATURE_NAMES]
        
        X_sub = X_combined[self.selected_features]
        return self.scaler.transform(X_sub)

    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        self.fit(X, y)
        X_trans = self.transform(X)
        
        if y is not None and self.use_smote:
            y_arr = np.array(y, dtype=int)
            classes, counts = np.unique(y_arr, return_counts=True)
            if len(classes) == 2 and min(counts) >= 6:
                k_neighbors = min(5, min(counts) - 1)
                smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
                X_res, y_res = smote.fit_resample(X_trans, y_arr)
                return X_res, y_res
            return X_trans, y_arr
        
        return X_trans, y

    def transform_single(self, patient_dict: Dict[str, Any]) -> np.ndarray:
        """Transforms a single clinical record dictionary into a 1xD normalized numpy vector."""
        df = pd.DataFrame([patient_dict])
        return self.transform(df)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> 'ClinicalDataPipeline':
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def load_raw_dataset(site_key: str) -> pd.DataFrame:
    """Loads a specific hospital's raw dataset with column names and binarized target."""
    info = HOSPITALS_INFO.get(site_key)
    if not info:
        raise ValueError(f"Unknown site key: {site_key}. Available: {list(HOSPITALS_INFO.keys())}")
    
    file_path = info['raw_file']
    cols = FEATURE_NAMES + [TARGET_NAME]
    df = pd.read_csv(file_path, names=cols, na_values='?')
    
    # Binarize target: 0 = healthy, 1+ = presence of heart disease
    df[TARGET_NAME] = (df[TARGET_NAME] > 0).astype(int)
    return df


def load_all_datasets() -> Dict[str, pd.DataFrame]:
    """Loads all 4 hospital datasets into a dictionary."""
    return {site: load_raw_dataset(site) for site in HOSPITALS_INFO.keys()}


def get_reference_uci_split(test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Returns train/test split of the reference Cleveland UCI dataset."""
    df = load_raw_dataset('cleveland')
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def get_federated_partitions(selected_features: Optional[List[str]] = None,
                             test_ratio: float = 0.2,
                             random_state: int = 42) -> Dict[str, Any]:
    """
    Prepares non-IID hospital client datasets and a global test set.
    Fits the pipeline ONCE on the reference Cleveland dataset so all clients
    share identical feature scaling and representation.
    """
    all_data = load_all_datasets()
    
    # Fit the reference pipeline on Cleveland training split
    X_clev = all_data['cleveland'][FEATURE_NAMES]
    y_clev = all_data['cleveland'][TARGET_NAME]
    X_clev_tr, X_clev_te, y_clev_tr, y_clev_te = train_test_split(
        X_clev, y_clev, test_size=test_ratio, random_state=random_state, stratify=y_clev
    )
    
    pipeline = ClinicalDataPipeline(selected_features=selected_features, use_smote=True)
    pipeline.fit(X_clev_tr)
    
    # Partition each hospital's data
    clients_data = {}
    global_test_X_list = []
    global_test_y_list = []
    
    for site_key, df in all_data.items():
        X = df[FEATURE_NAMES]
        y = df[TARGET_NAME]
        
        # Stratify if possible
        class_counts = y.value_counts()
        strat = y if (len(class_counts) > 1 and min(class_counts) >= 2) else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_ratio, random_state=random_state, stratify=strat
        )
        
        # Transform using the shared reference pipeline
        X_tr_trans = pipeline.transform(X_train)
        X_val_trans = pipeline.transform(X_val)
        
        # Apply SMOTE to local training partition if min class count >= 6
        y_train_arr = y_train.values
        classes, counts = np.unique(y_train_arr, return_counts=True)
        if len(classes) == 2 and min(counts) >= 6:
            k_neighbors = min(5, min(counts) - 1)
            smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
            X_tr_res, y_tr_res = smote.fit_resample(X_tr_trans, y_train_arr)
        else:
            X_tr_res, y_tr_res = X_tr_trans, y_train_arr
        
        clients_data[site_key] = {
            'site_key': site_key,
            'info': HOSPITALS_INFO[site_key],
            'X_train': X_tr_res,
            'y_train': y_tr_res,
            'X_val': X_val_trans,
            'y_val': y_val.values,
            'raw_train_count': len(X_train),
            'raw_val_count': len(X_val)
        }
        
        global_test_X_list.append(X_val_trans)
        global_test_y_list.append(y_val.values)
        
    global_test_X = np.vstack(global_test_X_list)
    global_test_y = np.concatenate(global_test_y_list)
    
    return {
        'clients': clients_data,
        'global_test': {
            'X_test': global_test_X,
            'y_test': global_test_y
        },
        'pipeline': pipeline
    }
