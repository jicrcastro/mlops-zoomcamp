import mlflow
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

def train_and_log_model(df):
    mlflow.set_experiment('yellow-taxi-march-2023')
    with mlflow.start_run():
        dv = DictVectorizer()
        train_dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
        X_train = dv.fit_transform(train_dicts)
        y_train = df['duration'].values

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        mlflow.log_param('model_type', 'LinearRegression')
        mlflow.log_metric('intercept', lr.intercept_)
        mlflow.sklearn.log_model(lr, artifact_path="model", registered_model_name="lr_yellow_taxi_march_2023")

        print(f"Model intercept: {lr.intercept_}")
        return dv, lr
