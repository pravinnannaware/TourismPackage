import os
import joblib
import pandas as pd
import mlflow
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
from pyngrok import ngrok
import subprocess

def train_model():
    """
    Loads prepared dataset splits, sets up a preprocessing pipeline with XGBoost,
    tunes hyperparameters using GridSearchCV, logs metrics/artifacts to MLflow,
    and exports the best model for Streamlit deployment.
    """

    # 1. Load train and test data splits generated during data prep
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze()
    ytest = pd.read_csv("ytest.csv").squeeze()

    # 2. Identify categorical and numerical feature columns
    categorical_features = Xtrain.select_dtypes(include=['object']).columns.tolist()
    numeric_features = Xtrain.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # 3. Create preprocessing steps (StandardScaler for numerical, OneHotEncoder for categorical)
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown="ignore"), categorical_features)
    )

    # 4. Handle class imbalance by calculating pos_weight
    class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
    model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

    pipeline = make_pipeline(preprocessor, model)

    # 5. Define hyperparameter tuning grid
    param_grid = {
        "xgbclassifier__n_estimators": [50, 100],
        "xgbclassifier__max_depth": [3, 5],
        "xgbclassifier__learning_rate": [0.05, 0.1]
    }

    # Set up Grid Search with 3-fold cross-validation optimizing for F1-score
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1", n_jobs=-1)
    

    # 6. MLflow Tracking and Model Training
    mlflow.set_experiment("Tourism_Package_Prediction")
    with mlflow.start_run():
        # Execute Grid Search
        grid.fit(Xtrain, ytrain)
        best_model = grid.best_estimator_
        
        # Evaluate model performance on test set
        preds = best_model.predict(Xtest)
        acc = accuracy_score(ytest, preds)
        f1 = f1_score(ytest, preds)

        # Log parameters and evaluation metrics to MLflow
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Print model performance summary
        print("=== Best Hyperparameters ===")
        print(grid.best_params_)
        print("\n=== Classification Report ===")
        print(classification_report(ytest, preds))

        # 7. Save best trained model artifact for deployment
        out_dir = "tourism_project/deployment/"
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, "best_tourism_model.joblib")
        joblib.dump(best_model, model_path)
        
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
