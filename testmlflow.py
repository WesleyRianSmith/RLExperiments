# Set mlflow
import mlflow
import pandas as pd
mlflow.set_experiment(experiment_name="wesleyriansmith-TunedPongPPO")
mlflow_client = mlflow.tracking.MlflowClient()


# Load your CSV data
data = pd.read_csv("ExperimentModels/PongPPO_Tuned/PongNoFrameskip-v4-PPO-2500000/metric_logs/progress.csv")


with mlflow.start_run():
    # Log parameters (e.g., hyperparameters of your model)
    #mlflow.log_param("param1", row["param1_column_name"])
    for index, row in data.iterrows():
        # Log metrics (e.g., accuracy, loss)
        if pd.notna(row['eval/mean_reward']):
            # Log metrics only if 'eval/mean_reward' is not NULL
            mlflow.log_metric("reward", row["eval/mean_reward"], step=int(row["time/total_timesteps"]))

            # Example print statement (corrected variable name)
            print(f"Logging {row['eval/mean_reward']} at timestep {row['time/total_timesteps']}")

            # You can also log artifacts (e.g., plots, model files) here if necessary
            # mlflow.log_artifact("path_to_your_artifact")
mlflow.start_run()


