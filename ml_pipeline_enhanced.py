"""
Enhanced Machine Learning Pipeline with Grid Search Logging
============================================================
This script provides a complete ML workflow with comprehensive tracking:
- Grid search parameter logging and visualization
- Feature tracking matrix (features x runs)
- Evaluation metrics visualization
- Detailed run metadata and comparison
- Training history and performance analysis

"""

import pandas as pd
import numpy as np
import json
import pickle
import os
import time
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import hashlib

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, ParameterGrid
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score
)

# Models
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS
# ============================================================================

top_features = [
    "잔액_리볼빙일시불이월_B0M",
    "월중평잔_RV일시불",
    "평잔_RV일시불_3M",
    "이용금액_신용_B0M",
    "잔액_신판평균한도소진율_r6m",
    "카드이용한도금액",
    "RV_최대잔액_R12M",
    "잔액_B0M"
]

# Dataset Configuration
TRAIN_DATASET_PATH = "artifacts/train_data.csv"
TEST_DATASET_PATH = "artifacts/test_data.csv"
TARGET_COLUMN = "잔액_리볼빙일시불이월_target"
FEATURE_COLUMNS = None

# ML Task Type
ML_TASK_TYPE = "regression"  # 'classification' or 'regression'
IS_BINARY_CLASSIFICATION = False

# Feature Filtering Configuration
CORRELATION_THRESHOLD = 0.8
REMOVE_OBJECT_COLUMNS = True
COLUMNS_TO_KEEP_EXPLICITLY = []

# Grid Search Configuration
GRID_SEARCH_ENABLED = True
GRID_SEARCH_PARAMS = {
  'random_forest': {
    'n_estimators': [400],
    'max_depth': [10],
    
    'min_samples_leaf': [10],
    'max_features' : [2,4,6,8,10,15],
    'bootstrap' : [True],
    'max_samples' : [0.6,0.8,1.0]
  }
}
#     },
#     'xgboost': {
#         'n_estimators': [200],
#         'max_depth': [10, 20],
#         'learning_rate': [0.01, 0.1],
#     },
#     'lightgbm': {
#         'n_estimators': [200],
#         'max_depth': [10, 20],
#         'learning_rate': [0.01, 0.1],
#     },
#     'gradient_boosting': {
#         'n_estimators': [200],
#         'max_depth': [10, 20],
#         'learning_rate': [0.01, 0.1]
#     }
# }

# Models to evaluate
MODELS_TO_EVALUATE = ['random_forest'] #'random_forest', 'xgboost', 'lightgbm'

# Hyperparameter Optimization Configuration
HYPERPARAM_OPTIMIZATION = True
N_ITER_SEARCH = 20
CV_FOLDS = 5
RANDOM_STATE = 123
N_JOBS = -1

# Data Optimization
SAMPLE_SIZE = None
USE_DOWNCASTING = True

# Output Configuration
OUTPUT_DIR = "analysis_enhanced"
GRID_SEARCH_LOG_DIR = "grid_search_logs"
VISUALIZATIONS_DIR = "visualizations"
LOG_FILENAME = "training.log"

# ============================================================================
# END OF CONFIGURATION SECTION
# ============================================================================


class GridSearchTracker:
    """Tracks and logs all grid search runs with detailed metrics"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.runs = []  # List of all runs
        self.run_counter = 0
        self.runs_dataframe = None
        self.feature_matrix = None
        self.metrics_history = []
        
        # Create subdirectories
        self.log_dir = self.output_dir / GRID_SEARCH_LOG_DIR
        self.viz_dir = self.output_dir / VISUALIZATIONS_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
    def log_run(self, 
                run_id: str,
                model_name: str,
                parameters: Dict[str, Any],
                features_used: List[str],
                metrics: Dict[str, float],
                training_time: float,
                cv_scores: List[float],
                memory_used: float,
                timestamp: str = None) -> None:
        """Log a single grid search run"""
        
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        run_record = {
            'run_id': run_id,
            'timestamp': timestamp,
            'model_name': model_name,
            'parameters': json.dumps(parameters),
            'n_features': len(features_used),
            'features_used': json.dumps(features_used),
            'training_time_seconds': training_time,
            'memory_used_mb': memory_used,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'cv_scores': json.dumps(cv_scores.tolist() if isinstance(cv_scores, np.ndarray) else cv_scores),
        }
        
        # Add metrics
        for metric_name, metric_value in metrics.items():
            run_record[f'test_{metric_name}'] = metric_value
        
        self.runs.append(run_record)
        self.run_counter += 1
        
        logger.info(f"Logged run {run_id}: {model_name} with {len(features_used)} features")
        
    def create_runs_dataframe(self) -> pd.DataFrame:
        """Convert runs list to DataFrame"""
        if not self.runs:
            logger.warning("No runs to create dataframe from")
            return pd.DataFrame()
        
        self.runs_dataframe = pd.DataFrame(self.runs)
        return self.runs_dataframe
    
    def create_feature_matrix(self, all_features: List[str]) -> pd.DataFrame:
        """Create feature usage matrix (features x runs)"""
        if not self.runs:
            logger.warning("No runs to create feature matrix from")
            return pd.DataFrame()
        
        feature_matrix = pd.DataFrame(0, 
                                     index=range(len(self.runs)), 
                                     columns=all_features)
        
        for idx, run in enumerate(self.runs):
            features = json.loads(run['features_used'])
            for feature in features:
                if feature in feature_matrix.columns:
                    feature_matrix.loc[idx, feature] = 1
        
        # Add run identifiers
        feature_matrix.insert(0, 'run_id', [run['run_id'] for run in self.runs])
        feature_matrix.insert(1, 'model_name', [run['model_name'] for run in self.runs])
        
        self.feature_matrix = feature_matrix
        return feature_matrix
    
    def save_runs_log(self) -> None:
        """Save runs to CSV and JSON"""
        if self.runs_dataframe is None:
            self.create_runs_dataframe()
        
        csv_path = self.log_dir / "grid_search_runs.csv"
        json_path = self.log_dir / "grid_search_runs.json"
        
        self.runs_dataframe.to_csv(csv_path, index=False)
        with open(json_path, 'w') as f:
            json.dump(self.runs, f, indent=2)
        
        logger.info(f"Saved runs log to {csv_path} and {json_path}")
    
    def save_feature_matrix(self) -> None:
        """Save feature matrix to CSV"""
        if self.feature_matrix is None:
            logger.warning("Feature matrix not created yet")
            return
        
        csv_path = self.log_dir / "feature_matrix.csv"
        self.feature_matrix.to_csv(csv_path, index=False)
        logger.info(f"Saved feature matrix to {csv_path}")
    
    def get_best_run(self, metric: str = 'test_mae') -> Optional[Dict]:
        """Get the best run based on a metric"""
        if self.runs_dataframe is None or self.runs_dataframe.empty:
            self.create_runs_dataframe()
        
        if metric not in self.runs_dataframe.columns:
            logger.warning(f"Metric {metric} not found in runs")
            return None
        
        # For MAE, MSE, lower is better
        if 'mae' in metric.lower() or 'mse' in metric.lower() or 'rmse' in metric.lower():
            best_idx = self.runs_dataframe[metric].idxmin()
        else:
            best_idx = self.runs_dataframe[metric].idxmax()
        
        return self.runs[best_idx]
    
    def get_runs_summary(self) -> pd.DataFrame:
        """Get summary of all runs"""
        if self.runs_dataframe is None or self.runs_dataframe.empty:
            self.create_runs_dataframe()
        
        summary_cols = ['run_id', 'model_name', 'n_features', 'training_time_seconds', 
                       'memory_used_mb', 'cv_mean', 'cv_std']
        
        # Add metric columns
        metric_cols = [col for col in self.runs_dataframe.columns if col.startswith('test_')]
        summary_cols.extend(metric_cols)
        
        available_cols = [col for col in summary_cols if col in self.runs_dataframe.columns]
        return self.runs_dataframe[available_cols]


class EnhancedMLPipeline:
    """Enhanced ML Pipeline with Grid Search Tracking"""
    
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_selected = None
        self.X_test_selected = None
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        self.label_encoder = None
        
        # Grid search tracking
        self.tracker = GridSearchTracker(OUTPUT_DIR)
        self.all_features = []
        self._custom_feature_columns = None  # For user-selected features
        
        # Create output directory
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        
        # Configure Loguru
        log_path = f"{OUTPUT_DIR}/{LOG_FILENAME}"
        logger.remove()
        logger.add(
            log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            mode="w",
            encoding="utf-8",
            level="INFO"
        )
        logger.add(
            lambda msg: tqdm.write(msg, end=""),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="INFO"
        )

    def select_columns_for_training(self, columns: List[str]):
        """
        Specify the columns you want to use for ML training.
        Call this method before run_pipeline() or load_data().
        """
        logger.info(f"Manually set columns for training: {columns}")
        self._custom_feature_columns = columns

    def load_data(self):
        """Load training and testing datasets"""
        logger.info("=" * 80)
        logger.info("STEP 1: LOADING DATA")
        logger.info("=" * 80)
        
        tqdm.write("Step 1: Loading data...")
        with tqdm(total=4, desc="Loading data", unit="step") as pbar:
            # Load training data
            logger.info(f"Loading training data from: {TRAIN_DATASET_PATH}")
            pbar.set_description("Loading training data")
            train_df = self._load_csv_optimized(TRAIN_DATASET_PATH)
            pbar.update(1)
            
            # Load testing data
            logger.info(f"Loading testing data from: {TEST_DATASET_PATH}")
            pbar.set_description("Loading test data")
            test_df = self._load_csv_optimized(TEST_DATASET_PATH)
            pbar.update(1)
            
            # Sample data if specified
            if SAMPLE_SIZE and len(train_df) > SAMPLE_SIZE:
                logger.info(f"Sampling {SAMPLE_SIZE} rows from training data")
                train_df = train_df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
            
            # Separate features and target
            pbar.set_description("Preparing features")
            # NEW: Allow user to specify feature columns, otherwise use config FEATURE_COLUMNS, otherwise default
            if self._custom_feature_columns is not None:
                logger.info(f"Using user-provided columns for training: {self._custom_feature_columns}")
                X_train = train_df[self._custom_feature_columns]
                X_test = test_df[self._custom_feature_columns]
                feature_columns_used = self._custom_feature_columns
            elif FEATURE_COLUMNS:
                logger.info(f"Using FEATURE_COLUMNS from config: {FEATURE_COLUMNS}")
                X_train = train_df[FEATURE_COLUMNS]
                X_test = test_df[FEATURE_COLUMNS]
                feature_columns_used = FEATURE_COLUMNS
            else:
                X_train = train_df.drop(columns=[TARGET_COLUMN])
                X_test = test_df.drop(columns=[TARGET_COLUMN])
                feature_columns_used = list(X_train.columns)
            
            y_train = train_df[TARGET_COLUMN]
            y_test = test_df[TARGET_COLUMN]
            
            # Handle categorical target
            if ML_TASK_TYPE == 'classification':
                if y_train.dtype == 'object':
                    self.label_encoder = LabelEncoder()
                    y_train = self.label_encoder.fit_transform(y_train)
                    y_test = self.label_encoder.transform(y_test)
            
            # Handle categorical features
            pbar.set_description("Encoding categorical features")
            X_train = self._encode_categorical_features(X_train)
            X_test = self._encode_categorical_features(X_test)
            pbar.update(1)
            
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            self.all_features = feature_columns_used
            pbar.update(1)
        
        logger.info(f"Training set shape: {self.X_train.shape}")
        logger.info(f"Testing set shape: {self.X_test.shape}")
        logger.info(f"Total features: {len(self.all_features)}")
        
    def _load_csv_optimized(self, filepath):
        """Load CSV with memory optimization"""
        sample = pd.read_csv(filepath, nrows=100)
        dtypes = {}
        
        for col in sample.columns:
            if sample[col].dtype == 'object':
                dtypes[col] = 'category'
            elif sample[col].dtype == 'float64':
                dtypes[col] = 'float32' if USE_DOWNCASTING else 'float64'
            elif sample[col].dtype == 'int64':
                dtypes[col] = 'int32' if USE_DOWNCASTING else 'int64'
        
        df = pd.read_csv(filepath, dtype=dtypes)
        return df
    
    def _encode_categorical_features(self, df):
        """Encode categorical features"""
        df = df.copy()
        for col in df.select_dtypes(include=['object', 'category']).columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        return df
    
    def preprocess_data(self):
        """Preprocess and scale data"""
        logger.info("=" * 80)
        logger.info("STEP 2: PREPROCESSING DATA")
        logger.info("=" * 80)
        
        # Handle missing values - remove columns with NaN
        logger.info("Checking for missing values...")
        
        # Convert to DataFrame if it's a numpy array (it should still be DataFrame at this point)
        if isinstance(self.X_train, np.ndarray):
            X_train_df = pd.DataFrame(self.X_train, columns=self.all_features)
            X_test_df = pd.DataFrame(self.X_test, columns=self.all_features)
        else:
            X_train_df = self.X_train.copy()
            X_test_df = self.X_test.copy()
        
        # Identify columns with NaN values
        nan_cols_train = X_train_df.columns[X_train_df.isna().any()].tolist()
        nan_cols_test = X_test_df.columns[X_test_df.isna().any()].tolist()
        nan_cols = list(set(nan_cols_train + nan_cols_test))
        
        if nan_cols:
            logger.warning(f"Found {len(nan_cols)} columns with NaN values: {nan_cols[:10]}{'...' if len(nan_cols) > 10 else ''}")
            
            # Log detailed NaN statistics
            nan_stats = []
            for col in nan_cols[:20]:  # Log first 20 columns
                train_nan_count = X_train_df[col].isna().sum()
                test_nan_count = X_test_df[col].isna().sum() if col in X_test_df.columns else 0
                train_nan_pct = (train_nan_count / len(X_train_df)) * 100
                test_nan_pct = (test_nan_count / len(X_test_df)) * 100 if col in X_test_df.columns else 0
                nan_stats.append(f"  {col}: Train {train_nan_count} ({train_nan_pct:.2f}%), Test {test_nan_count} ({test_nan_pct:.2f}%)")
            
            logger.info("NaN statistics (showing first 20 columns):")
            for stat in nan_stats:
                logger.info(stat)
            
            # Drop columns with NaN
            X_train_df = X_train_df.drop(columns=nan_cols)
            X_test_df = X_test_df.drop(columns=nan_cols)
            
            # Update feature list
            self.all_features = [f for f in self.all_features if f not in nan_cols]
            
            logger.info(f"Dropped {len(nan_cols)} columns with NaN values")
            logger.info(f"Remaining features: {len(self.all_features)}")
        else:
            logger.info("No missing values found in the data")
        
        # Check for NaN in target variable
        if pd.Series(self.y_train).isna().any():
            logger.warning("Found NaN values in training target variable - removing affected rows")
            valid_idx = ~pd.Series(self.y_train).isna()
            X_train_df = X_train_df[valid_idx]
            self.y_train = self.y_train[valid_idx]
            logger.info(f"Remaining training samples: {len(self.y_train)}")
        
        if pd.Series(self.y_test).isna().any():
            logger.warning("Found NaN values in test target variable - removing affected rows")
            valid_idx = ~pd.Series(self.y_test).isna()
            X_test_df = X_test_df[valid_idx]
            self.y_test = self.y_test[valid_idx]
            logger.info(f"Remaining test samples: {len(self.y_test)}")
        
        # Convert back to numpy array for scaling
        self.X_train = X_train_df.values
        self.X_test = X_test_df.values
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        logger.info("Data scaled using StandardScaler")
        logger.info(f"Final training shape: {self.X_train.shape}")
        logger.info(f"Final test shape: {self.X_test.shape}")
    
    def select_features(self):
        """Skip feature selection and use all features"""
        logger.info("=" * 80)
        logger.info("STEP 3: FEATURE SELECTION (DISABLED)")
        logger.info("=" * 80)
        
        # Use all features without selection
        self.selected_features = self.all_features
        self.X_train_selected = self.X_train
        self.X_test_selected = self.X_test
        
        logger.info(f"Feature selection is disabled - using all {len(self.all_features)} features")
        logger.info(f"Training shape: {self.X_train_selected.shape}")
        logger.info(f"Test shape: {self.X_test_selected.shape}")
    
    def run_grid_search(self):
        """Run grid search with tracking"""
        logger.info("=" * 80)
        logger.info("STEP 4: GRID SEARCH WITH TRACKING")
        logger.info("=" * 80)
        
        if not GRID_SEARCH_ENABLED:
            logger.info("Grid search disabled")
            return
        
        X_train = self.X_train_selected if self.X_train_selected is not None else self.X_train
        X_test = self.X_test_selected if self.X_test_selected is not None else self.X_test
        features = self.selected_features if self.selected_features else self.all_features
        
        total_combinations = 0
        for model_name in MODELS_TO_EVALUATE:
            if model_name in GRID_SEARCH_PARAMS:
                param_grid = GRID_SEARCH_PARAMS[model_name]
                combinations = len(list(ParameterGrid(param_grid)))
                total_combinations += combinations
        
        logger.info(f"Total parameter combinations to test: {total_combinations}")
        
        with tqdm(total=total_combinations, desc="Grid Search Progress") as pbar:
            for model_name in MODELS_TO_EVALUATE:
                if model_name not in GRID_SEARCH_PARAMS:
                    logger.warning(f"No parameters defined for {model_name}")
                    continue
                
                self._run_grid_search_for_model(
                    model_name, 
                    X_train, X_test, 
                    features, 
                    pbar
                )
    
    def _run_grid_search_for_model(self, 
                                   model_name: str, 
                                   X_train: np.ndarray, 
                                   X_test: np.ndarray,
                                   features: List[str],
                                   pbar: tqdm) -> None:
        """Run grid search for a specific model"""
        
        model_class = self._get_model_class(model_name)
        param_grid = GRID_SEARCH_PARAMS[model_name]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Grid Search for {model_name.upper()}")
        logger.info(f"{'='*60}")
        
        for params in ParameterGrid(param_grid):
            run_id = self._generate_run_id(model_name, params)
            
            try:
                # Record start time and memory
                start_time = time.time()
                process = psutil.Process()
                mem_start = process.memory_info().rss / 1024 / 1024  # MB
                
                # Create and train model
                model = model_class(random_state=RANDOM_STATE, n_jobs=N_JOBS, **params)
                
                # Cross-validation scores
                cv_scores = cross_val_score(
                    model, X_train, self.y_train, 
                    cv=CV_FOLDS, 
                    scoring=self._get_scoring_metric()
                )
                
                # Train on full training set
                model.fit(X_train, self.y_train)
                
                # Evaluate on test set
                y_pred = model.predict(X_test)
                metrics = self._calculate_metrics(self.y_test, y_pred, model_name)
                
                # Record end time and memory
                training_time = time.time() - start_time
                mem_end = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = mem_end - mem_start
                
                # Log the run
                self.tracker.log_run(
                    run_id=run_id,
                    model_name=model_name,
                    parameters=params,
                    features_used=features,
                    metrics=metrics,
                    training_time=training_time,
                    cv_scores=cv_scores,
                    memory_used=memory_used
                )
                
                logger.info(f"✓ Run {run_id} completed | CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                logger.error(f"✗ Run {run_id} failed: {str(e)}")
            
            pbar.update(1)
    
    def _generate_run_id(self, model_name: str, params: Dict) -> str:
        """Generate unique run ID"""
        param_str = '_'.join([f"{k}={v}" for k, v in sorted(params.items())])
        hash_obj = hashlib.md5(param_str.encode())
        return f"{model_name}_{self.tracker.run_counter}_{hash_obj.hexdigest()[:8]}"
    
    def _get_model_class(self, model_name: str):
        """Get model class based on name"""
        models = {
            'ridge': Ridge,
            'lasso': Lasso,
            'random_forest': RandomForestRegressor if ML_TASK_TYPE == 'regression' else RandomForestClassifier,
            'gradient_boosting': GradientBoostingRegressor if ML_TASK_TYPE == 'regression' else GradientBoostingClassifier,
            'xgboost': XGBRegressor if ML_TASK_TYPE == 'regression' else XGBClassifier,
            'lightgbm': LGBMRegressor if ML_TASK_TYPE == 'regression' else LGBMClassifier,
            'svr': SVR,
            'svm': SVC,
            'knn': KNeighborsRegressor if ML_TASK_TYPE == 'regression' else KNeighborsClassifier,
            'logistic': LogisticRegression,
        }
        return models.get(model_name, RandomForestRegressor)
    
    def _get_scoring_metric(self) -> str:
        """Get scoring metric based on task type"""
        if ML_TASK_TYPE == 'regression':
            return 'neg_mean_absolute_error'
        else:
            return 'f1' if IS_BINARY_CLASSIFICATION else 'f1_weighted'
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        metrics = {}
        
        if ML_TASK_TYPE == 'regression':
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_true, y_pred)
        else:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return metrics
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        logger.info("=" * 80)
        logger.info("STEP 5: GENERATING VISUALIZATIONS")
        logger.info("=" * 80)
        
        if not self.tracker.runs:
            logger.warning("No runs to visualize")
            return
        
        self.tracker.create_runs_dataframe()
        self.tracker.create_feature_matrix(self.all_features)
        
        # Generate visualizations
        self._plot_metrics_comparison()
        self._plot_model_comparison()
        self._plot_training_time_vs_performance()
        self._plot_feature_usage()
        self._plot_cv_scores_distribution()
        self._plot_runs_summary_table()
        
        logger.info("Visualizations generated successfully")
    
    def _plot_metrics_comparison(self):
        """Plot metrics comparison across runs"""
        df = self.tracker.runs_dataframe
        metric_cols = [col for col in df.columns if col.startswith('test_')]
        
        if not metric_cols:
            return
        
        fig, axes = plt.subplots(len(metric_cols), 1, figsize=(14, 4 * len(metric_cols)))
        if len(metric_cols) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metric_cols):
            ax = axes[idx]
            data = df.sort_values(metric)
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(data)))
            
            ax.barh(range(len(data)), data[metric], color=colors)
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels([f"{row['model_name']}_{i}" for i, (_, row) in enumerate(data.iterrows())])
            ax.set_xlabel(metric)
            ax.set_title(f'{metric.upper()} Across All Runs')
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.tracker.viz_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved metrics_comparison.png")
    
    def _plot_model_comparison(self):
        """Plot comparison of different models"""
        df = self.tracker.runs_dataframe
        metric_cols = [col for col in df.columns if col.startswith('test_')]
        
        if not metric_cols:
            return
        
        fig, axes = plt.subplots(1, len(metric_cols), figsize=(5 * len(metric_cols), 6))
        if len(metric_cols) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metric_cols):
            ax = axes[idx]
            model_data = df.groupby('model_name')[metric].agg(['mean', 'std']).reset_index()
            
            x_pos = np.arange(len(model_data))
            ax.bar(x_pos, model_data['mean'], yerr=model_data['std'], 
                   capsize=5, alpha=0.7, color='steelblue')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_data['model_name'], rotation=45)
            ax.set_ylabel(metric)
            ax.set_title(f'{metric.upper()} by Model')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.tracker.viz_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved model_comparison.png")
    
    def _plot_training_time_vs_performance(self):
        """Plot training time vs performance"""
        df = self.tracker.runs_dataframe
        metric_cols = [col for col in df.columns if col.startswith('test_')]
        
        if not metric_cols:
            return
        
        fig, axes = plt.subplots(1, len(metric_cols), figsize=(6 * len(metric_cols), 5))
        if len(metric_cols) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metric_cols):
            ax = axes[idx]
            models = df['model_name'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
            
            for color, model in zip(colors, models):
                model_df = df[df['model_name'] == model]
                ax.scatter(model_df['training_time_seconds'], model_df[metric], 
                          label=model, alpha=0.6, s=100, color=color)
            
            ax.set_xlabel('Training Time (seconds)')
            ax.set_ylabel(metric)
            ax.set_title(f'Training Time vs {metric.upper()}')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.tracker.viz_dir / 'training_time_vs_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved training_time_vs_performance.png")
    
    def _plot_feature_usage(self):
        """Plot feature usage heatmap"""
        if self.tracker.feature_matrix is None:
            return
        
        # Create heatmap of feature usage
        feature_cols = [col for col in self.tracker.feature_matrix.columns 
                       if col not in ['run_id', 'model_name']]
        heatmap_data = self.tracker.feature_matrix[feature_cols].astype(int)
        
        fig, ax = plt.subplots(figsize=(16, max(8, len(self.tracker.feature_matrix) * 0.3)))
        
        sns.heatmap(heatmap_data, cmap='YlGn', cbar_kws={'label': 'Feature Used'}, ax=ax)
        ax.set_yticklabels([f"{row['model_name']}_{i}" 
                           for i, (_, row) in enumerate(self.tracker.feature_matrix.iterrows())],
                          rotation=0, fontsize=8)
        ax.set_xticklabels(feature_cols, rotation=45, ha='right', fontsize=8)
        ax.set_title('Feature Usage Across Runs')
        
        plt.tight_layout()
        plt.savefig(self.tracker.viz_dir / 'feature_usage_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved feature_usage_heatmap.png")
    
    def _plot_cv_scores_distribution(self):
        """Plot cross-validation scores distribution"""
        df = self.tracker.runs_dataframe
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        cv_data = []
        labels = []
        for idx, row in df.iterrows():
            cv_scores = json.loads(row['cv_scores'])
            cv_data.append(cv_scores)
            labels.append(f"{row['model_name']}_{idx}")
        
        bp = ax.boxplot(cv_data, labels=labels)
        ax.set_ylabel('Cross-Validation Score')
        ax.set_title('Cross-Validation Scores Distribution')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.tracker.viz_dir / 'cv_scores_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved cv_scores_distribution.png")
    
    def _plot_runs_summary_table(self):
        """Create and save runs summary as table visualization"""
        summary_df = self.tracker.get_runs_summary()
        
        # Create figure with table
        fig, ax = plt.subplots(figsize=(16, max(8, len(summary_df) * 0.4)))
        ax.axis('tight')
        ax.axis('off')
        
        # Format numeric columns
        display_df = summary_df.copy()
        for col in display_df.columns:
            if col not in ['run_id', 'model_name']:
                if display_df[col].dtype in ['float64', 'float32']:
                    display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
        
        table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        # Color header
        for i in range(len(display_df.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(display_df) + 1):
            for j in range(len(display_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('Grid Search Runs Summary', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(self.tracker.viz_dir / 'runs_summary_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved runs_summary_table.png")
    
    def save_all_logs(self):
        """Save all tracking logs"""
        logger.info("=" * 80)
        logger.info("STEP 6: SAVING LOGS AND REPORTS")
        logger.info("=" * 80)
        
        self.tracker.save_runs_log()
        self.tracker.save_feature_matrix()
        
        # Save best run info
        best_run = self.tracker.get_best_run()
        if best_run:
            best_run_path = self.tracker.log_dir / "best_run.json"
            with open(best_run_path, 'w') as f:
                json.dump(best_run, f, indent=2)
            logger.info(f"Saved best run to {best_run_path}")
        
        # Generate summary report
        self._generate_summary_report()
    
    def _generate_summary_report(self):
        """Generate comprehensive summary report"""
        report_path = self.tracker.log_dir / "summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("GRID SEARCH SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total Runs: {len(self.tracker.runs)}\n")
            f.write(f"Total Features: {len(self.all_features)}\n")
            f.write(f"Selected Features: {len(self.selected_features) if self.selected_features else len(self.all_features)}\n\n")
            
            # Best run
            best_run = self.tracker.get_best_run()
            if best_run:
                f.write("BEST RUN:\n")
                f.write(f"  Run ID: {best_run['run_id']}\n")
                f.write(f"  Model: {best_run['model_name']}\n")
                f.write(f"  Parameters: {best_run['parameters']}\n")
                f.write(f"  CV Score: {best_run['cv_mean']:.4f} ± {best_run['cv_std']:.4f}\n")
                f.write(f"  Training Time: {best_run['training_time_seconds']:.2f}s\n")
                f.write(f"  Memory Used: {best_run['memory_used_mb']:.2f}MB\n\n")
            
            # Model statistics
            f.write("MODEL STATISTICS:\n")
            model_stats = self.tracker.runs_dataframe.groupby('model_name').agg({
                'training_time_seconds': ['mean', 'min', 'max'],
                'memory_used_mb': ['mean', 'min', 'max'],
                'cv_mean': ['mean', 'std']
            }).round(4)
            f.write(model_stats.to_string())
            f.write("\n\n")
            
            # Runs summary
            f.write("RUNS SUMMARY:\n")
            summary = self.tracker.get_runs_summary()
            f.write(summary.to_string())
        
        logger.info(f"Saved summary report to {report_path}")
    
    def run_pipeline(self):
        """Run complete pipeline"""
        try:
            self.load_data()
            self.preprocess_data()
            self.select_features()
            self.run_grid_search()
            self.generate_visualizations()
            self.save_all_logs()
            
            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"Output directory: {OUTPUT_DIR}")
            logger.info(f"Grid search logs: {self.tracker.log_dir}")
            logger.info(f"Visualizations: {self.tracker.viz_dir}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise


def main():
    """Main entry point"""
    pipeline = EnhancedMLPipeline()
    # Example usage: pipeline.select_columns_for_training(['col1','col2','col3'])
    pipeline.select_columns_for_training(top_features)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
