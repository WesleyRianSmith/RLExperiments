# Set mlflow
import mlflow
import pandas as pd
#mlflow.set_tracking_uri("http://seventheli-mlflow.eu.cpolar.io")
mlflow.set_experiment(experiment_name="wesleyriansmith-BreakoutPPO")
mlflow_client = mlflow.tracking.MlflowClient()


# Load your CSV data
data = pd.read_csv("test.csv")


with mlflow.start_run():
    # Log parameters (e.g., hyperparameters of your model)
    #mlflow.log_param("param1", row["param1_column_name"])
    for index, row in data.iterrows():
        # Log metrics (e.g., accuracy, loss)
        mlflow.log_metric("reward", row["rollout/ep_rew_mean"], step=int(row["time/total_timesteps"]))

        print(f"logging {row['rollout/ep_rew_mean']}")

        # You can also log artifacts (e.g., plots, model files)
        # mlflow.log_artifact("path_to_your_artifact")
mlflow.start_run()