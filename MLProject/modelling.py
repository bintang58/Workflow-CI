import os
import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

def load_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Data file not found at {data_path}")
    return pd.read_csv(data_path)

def setup_mlflow():
    dagshub_uri = "https://dagshub.com/bintang58/diabetes-prediction-model.mlflow"
    if os.getenv('DAGSHUB_TOKEN') and os.getenv('DAGSHUB_USERNAME'):
        os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('DAGSHUB_USERNAME')
        os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('DAGSHUB_TOKEN')
        mlflow.set_tracking_uri(dagshub_uri)
        mlflow.set_experiment("logistic_regression_experiment")
        print("✅ Using remote MLflow tracking on DagsHub")
        return True
    else:
        os.makedirs('mlruns', exist_ok=True)
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("logistic_regression_experiment_local")
        print("⚠️ MLflow tracking fallback to local: ./mlruns")
        return False

def start_run_with_retry(run_name=None, max_retries=5, wait_seconds=5):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"🔄 Attempt {attempt} to start MLflow run...")
            return mlflow.start_run(run_name=run_name)
        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(wait_seconds)
            else:
                raise e

def register_model_with_retry(model_uri, model_name, max_retries=5, wait_seconds=5):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"🔄 Attempt {attempt} to register model...")
            result = mlflow.register_model(model_uri=model_uri, name=model_name)
            print("✅ Model registered successfully!")
            return result
        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(wait_seconds)
            else:
                raise e

def main(args):
    use_remote_tracking = setup_mlflow()

    df = load_data(args.data_path)
    X = df.drop(columns=['diabetes'])
    y = df['diabetes']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    print(f"📊 Training set: {X_train.shape}, Test set: {X_test.shape}")

    mlflow.sklearn.autolog()

    with start_run_with_retry(run_name="LogisticRegression_GridSearchCV") as run:
        run_id = run.info.run_id
        print(f"🚀 MLflow Run ID: {run_id}")

        # GridSearchCV Logistic Regression
        param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs'],
        }
        grid = GridSearchCV(
            LogisticRegression(random_state=args.random_state),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_

        print(f"🔍 Best Params: {grid.best_params_}")
        mlflow.log_params(grid.best_params_)  # log hasil GridSearchCV

        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        mlflow.log_metrics({
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall
        })

        # Log params tambahan
        mlflow.log_params({
            "model_type": "LogisticRegression (GridSearchCV)",
            "test_size": args.test_size,
            "random_state": args.random_state
        })

        # Classification report
        report = classification_report(y_test, y_pred)
        report_path = "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        # Save Model
        os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
        joblib.dump(model, args.model_output)
        mlflow.log_artifact(args.model_output)
        print(f"✅ Model saved at: {args.model_output}")

        # MLflow Model Registry
        model_uri = f"runs:/{run_id}/model"
        model_name = "diabetes-prediction-model"
        try:
            register_model_with_retry(model_uri=model_uri, model_name=model_name)
        except Exception as error:
            print(f"❌ Model registry failed: {error}")

        if use_remote_tracking:
            print("\n🌐 MLflow Tracking on DagsHub:")
            print("🔗 https://dagshub.com/bintang58/diabetes-prediction-model.mlflow")
            print(f"mlflow models serve -m 'models:/{model_name}/latest' --port 5000")
        else:
            print("\n💻 Serve Locally:")
            print(f"mlflow models serve -m '{model_uri}' --port 5000")

        # Cleanup
        for file in [report_path, cm_path]:
            if os.path.exists(file):
                os.remove(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Logistic Regression with GridSearchCV for Diabetes Prediction")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--model_output", type=str, required=True)
    args = parser.parse_args()

    main(args)