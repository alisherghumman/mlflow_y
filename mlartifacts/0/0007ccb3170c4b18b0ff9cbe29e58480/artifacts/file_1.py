import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000")

import mlflow
client = mlflow.tracking.MlflowClient()

print("\n--- EXPERIMENTS SEEN BY SERVER ---")
for exp in client.search_experiments():
    print(f"ID: {exp.experiment_id}, Name: {exp.name}")
print("----------------------------------\n")

if mlflow.active_run():
    mlflow.end_run()

mlflow.autolog(disable=True)
mlflow.sklearn.autolog(disable=True)



# client = mlflow.tracking.MlflowClient()

# print("\nEXPERIMENTS SEEN BY PYTHON:\n")
# for exp in client.search_experiments():
#     print(exp.experiment_id, exp.name)


wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

max_depth = 10
n_estimators = 10


client = mlflow.tracking.MlflowClient()

print("\n--- EXPERIMENTS SEEN BY SERVER ---")
for exp in client.search_experiments():
    print(f"ID: {exp.experiment_id}, Name: {exp.name}")
print("----------------------------------\n")

current_experiment = mlflow.set_experiment("mlflow-y")

with mlflow.start_run(experiment_id=current_experiment.experiment_id):
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    plt.savefig("Confusion_matrix.png")

    mlflow.log_artifact("Confusion_matrix.png")
    mlflow.log_artifact(__file__)

    print(accuracy)