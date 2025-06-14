import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import xgboost_model as xgb
import warnings

warnings.filterwarnings('ignore')

# --- 1. Load and Prepare Data ---
DATA_FILE = "../data/augmented_data_by_noise.csv"

try:
    df = pd.read_csv(DATA_FILE)
    print(f"✅ Data loaded successfully from '{DATA_FILE}'. Shape: {df.shape}")
except FileNotFoundError:
    print(f"❌ Error: Data file '{DATA_FILE}' not found. Please check the path.")
    exit()

# Define features (X) and target (y)
TARGET_COLUMN = 'Glucose_classification_class'

# Define the features to be used for training, excluding the direct glucose reading and the target
FEATURES = [
    'Age', 'Systolic_BP', 'Diastolic_BP', 'BMI', 'Pulse rate',
    'HbA1c_%', 'Respiratory Rate', 'Acetone PPM 1.1', 'Temperature', 'Humidity'
]

# Ensure all specified features exist in the dataframe
existing_features = [f for f in FEATURES if f in df.columns]
if len(existing_features) != len(FEATURES):
    print("⚠️ Warning: Some features were not found in the dataframe.")
    FEATURES = existing_features

X = df[FEATURES]
y = df[TARGET_COLUMN]

# Encode the target variable (e.g., 'Low', 'Normal', 'High' -> 0, 1, 2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nTarget classes encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# --- 2. Define Hyperparameter Search Space for XGBoost ---
# These are the ranges Bayesian Optimization will search within.
search_space = [
    Integer(100, 1000, name='n_estimators'),
    Real(0.01, 0.3, 'log-uniform', name='learning_rate'),
    Integer(3, 10, name='max_depth'),
    Real(0.1, 1.0, name='subsample'),
    Real(0.1, 1.0, name='colsample_bytree')
]

# --- 3. Define the Objective Function for Bayesian Optimization ---
# This function takes a set of hyperparameters, trains a model using cross-
# validation, and returns a score for the optimizer to minimize.
# We return a negative score because the optimizer minimizes, and we want to maximize F1-score.

# Set up K-Fold cross-validation
N_SPLITS = 5  # Using 5 folds
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


@use_named_args(search_space)
def objective(**params):
    """
    Objective function for Bayesian Optimization.
    Trains an XGBoost model with given params using cross-validation.
    """
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        **params
    )

    # This list will store the f1_score of each fold
    fold_scores = []

    # Loop through each fold
    for train_index, val_index in skf.split(X, y_encoded):
        # Split data into training and validation for this fold
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y_encoded[train_index], y_encoded[val_index]

        # Create a pipeline that first applies SMOTE then trains the model
        # SMOTE is applied ONLY to the training data of this specific fold
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])

        # Train the pipeline
        pipeline.fit(X_train, y_train)

        # Make predictions on the untouched validation set
        y_pred = pipeline.predict(X_val)

        # Calculate the F1 score for this fold (weighted to handle class imbalance)
        score = f1_score(y_val, y_pred, average='weighted')
        fold_scores.append(score)

    # Return the negative average score across all folds
    average_score = np.mean(fold_scores)

    # Print progress
    print(f"Params: {params} -> Avg F1-score: {-average_score:.4f}")

    return -average_score


# --- 4. Run Bayesian Optimization ---
# n_calls is the number of different hyperparameter sets to try.
# More calls can lead to better results but will take longer.
print("\n🚀 Starting Bayesian Optimization to find the best XGBoost hyperparameters...")
print("This will take some time...")

gp_result = gp_minimize(
    func=objective,
    dimensions=search_space,
    n_calls=50,  # Try 50 different combinations
    random_state=42,
    n_jobs=-1  # Use all available CPU cores
)

# --- 5. Display Results ---
print("\n🎉 Bayesian Optimization Finished!")
print(f"Best F1-score (weighted): {-gp_result.fun:.4f}")
print("\nBest parameters found:")
best_params = dict(zip([s.name for s in search_space], gp_result.x))
for param, value in best_params.items():
    print(f"  - {param}: {value}")

print("\nThese are the optimal hyperparameters to use for training your final model.")

