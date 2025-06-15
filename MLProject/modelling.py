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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    classification_report, confusion_matrix
)

def load_data(data_path):
    """Load dataset from the specified path."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    return pd.read_csv(data_path)

def setup_mlflow():
    """Setup MLflow tracking, remote (DagsHub) or local."""
    if os.environ.get('DAGSHUB_TOKEN') and os.environ.get('DAGSHUB_USERNAME'):
        os.environ['MLFLOW_TRACKING_USERNAME'] = os.environ['DAGSHUB_USERNAME']
        os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ['DAGSHUB_TOKEN']
        mlflow.set_tracking_uri("https://dagshub.com/bintang58/diabetes-prediction-model.mlflow")
        mlflow.set_experiment("diabetes-prediction-experiment")
        print("✅ Using remote MLflow tracking on DagsHub")
        return True
    else:
        os.makedirs('mlruns', exist_ok=True)
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("diabetes-prediction-experiment-local")
        print("✅ MLflow tracking tersimpan secara lokal di: ./mlruns")
        return False

def register_model_with_retry(model_uri, model_name, max_retries=5, wait_seconds=5):
    """Register MLflow model with retry mechanism."""
    for attempt in range(max_retries):
        try:
            print(f"🔄 Attempt {attempt + 1} to register model...")
            result = mlflow.register_model(model_uri=model_uri, name=model_name)
            print("✅ Model registered successfully!")
            return result
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)
            else:
                print("❌ Maximum retry attempts reached. Model registration failed.")
                raise e

def main(args):
    # Setup MLflow tracking
    use_remote_tracking = setup_mlflow()

    # Load dataset
    df = load_data(args.data_path)
    X = df.drop(columns=['diabetes'])
    y = df['diabetes']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    print(f"📊 Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Train model
        model = LogisticRegression(random_state=args.random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Log params
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)

        # Classification report
        report = classification_report(y_test, y_pred)
        report_path = "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # Confusion matrix
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

        # Save model manually
        os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
        joblib.dump(model, args.model_output)
        mlflow.log_artifact(args.model_output)
        print(f"✅ Model disimpan ke: {args.model_output}")

        # Registrasi model ke MLflow Model Registry dengan retry
        model_path_uri = f"runs:/{run_id}/model"
        model_registry_name = "diabetes-prediction-model"

        # Cek apakah di environment CI (GitHub Actions)
        in_ci = os.getenv('GITHUB_ACTIONS') == 'true'

        if not in_ci:
            try:
                register_model_with_retry(model_uri=model_path_uri, model_name=model_registry_name)
            except Exception as error:
                print(f"❌ Registrasi model gagal meskipun sudah di-retry: {error}")
        else:
            print("⚠️ Skipping model registration in CI environment to avoid DagsHub errors.")

        # Informasi untuk serving
        if use_remote_tracking:
            print("\n🌐 Tracking MLflow tersedia di DagsHub:")
            print("🔗 https://dagshub.com/bintang58/diabetes-prediction-model.mlflow")
            print("🚀 Jalankan perintah berikut untuk serving model dari DagsHub:")
            print(f" mlflow models serve -m 'models:/{model_registry_name}/latest' --port 5000")
        else:
            print("\n💻 Jalankan perintah berikut untuk serving model secara lokal:")
            print(f"mlflow models serve -m '{model_path_uri}' --port 5000")

        # Clean up artifacts
        for file in [report_path, cm_path]:
            if os.path.exists(file):
                os.remove(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Logistic Regression untuk prediksi diabetes")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--model_output", type=str, required=True)
    args = parser.parse_args()

    main(args)