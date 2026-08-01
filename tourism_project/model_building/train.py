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


def train_model():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze()
    ytest = pd.read_csv("ytest.csv").squeeze()

    categorical_features = Xtrain.select_dtypes(include=['object']).columns.tolist()
    numeric_features = Xtrain.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown="ignore"), categorical_features)
    )

    # Class balance weighting
    class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
    model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

    pipeline = make_pipeline(preprocessor, model)

    param_grid = {
        "xgbclassifier__n_estimators": [50, 100],
        "xgbclassifier__max_depth": [3, 5],
        "xgbclassifier__learning_rate": [0.05, 0.1]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1", n_jobs=-1)
    # Set your auth token here (replace with your actual token)
    ngrok.set_auth_token("3H5xZmzJEEmUbSvirfMjxre110G_3dVZzySUQAKg9PWtGzEk5")

    # Start MLflow UI on port 5000
    process = subprocess.Popen(["mlflow", "ui", "--port", "5000"])

    # Create public tunnel
    public_url = ngrok.connect(5000).public_url
    print("MLflow UI is available at:", public_url)
    mlflow.set_tracking_uri(public_url)
    mlflow.set_experiment("Tourism_Package_Prediction")
    with mlflow.start_run():
        grid.fit(Xtrain, ytrain)
        best_model = grid.best_estimator_
        
        preds = best_model.predict(Xtest)
        acc = accuracy_score(ytest, preds)
        f1 = f1_score(ytest, preds)

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        print("=== Best Hyperparameters ===")
        print(grid.best_params_)
        print("\n=== Classification Report ===")
        print(classification_report(ytest, preds))

        # Save model for Streamlit deployment
        out_dir = "tourism_project/deployment/"
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, "best_tourism_model.joblib")
        joblib.dump(best_model, model_path)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
