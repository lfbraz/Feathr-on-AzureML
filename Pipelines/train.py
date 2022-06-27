import config
import os
import pickle

from feathr import FeathrClient
from feathr import HdfsSource
from feathr import TypedKey, ValueType, Feature, FeatureAnchor, INT32, INPUT_CONTEXT
from feathr import FeatureQuery, ObservationSettings
from feathr import TypedKey, ValueType, INT32
from feathr import SparkExecutionConfiguration
from feathr.job_utils import get_result_df

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import numpy as np

from azureml.core.run import Run
from azureml.core.model import Model

run = Run.get_context()

def build_features():
    client = FeathrClient(config_path='feathr_config.yaml')

    batch_source = HdfsSource(name="Customer",
                          path="dbfs:/delta/Calls/")

    customer_id = TypedKey(key_column="CustomerId",
                        key_column_type=ValueType.INT32,
                        description="CustomerId",
                        full_name="CustomerId")

    features = [
        Feature(name="f_NumberOfCalls",
                feature_type=INT32,
                key=customer_id,
                transform="NumberOfCalls"),
        Feature(name="f_AverageCallDuration",
                feature_type=INT32,
                key=customer_id,
                transform="AverageCallDuration"),
    ]

    request_anchor = FeatureAnchor(name="request_features",
                                source=batch_source,
                                features=features)


    client.build_features(anchor_list=[request_anchor])

    output_path = 'dbfs:/feathrazure_output'

    feature_query = FeatureQuery(
        feature_list=["f_NumberOfCalls", "f_AverageCallDuration"], key=customer_id)

    settings = ObservationSettings(
        observation_path="dbfs:/delta/Customer/")

    client.get_offline_features(observation_settings=settings,
                                feature_query=feature_query,
                                output_path=output_path,
                                execution_configuratons=SparkExecutionConfiguration({"spark.feathr.inputFormat": "delta", 
                                                                                    "spark.feathr.outputFormat": "delta"}))
    client.wait_job_to_finish(timeout_sec=500)

    result = get_result_df(client, format="delta", res_url = output_path)
    return result

def persist_model(model):
    os.makedirs('outputs', exist_ok=True)
    pickle.dump(model, open("outputs/model.pickle", "wb"))

def train_model(dataset):
    seed = 2022
    target_column = 'Churn'

    columns = ['CustomerId', 'NumberOfLoans', 'SatisfactionIndex', 'ActualBalance', 'CLTV', 'f_NumberOfCalls', 'f_AverageCallDuration', 'Churn']

    customer_train_dataset = dataset[columns]

    train, test = train_test_split(customer_train_dataset, random_state=seed, test_size=0.33)

    drop_columns = [target_column, 'CustomerId'] 

    X_train = train.drop(drop_columns, axis=1)
    X_test = test.drop(drop_columns, axis=1)

    y_train = train[target_column]
    y_test = test[target_column]

    n_estimators = 3

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(seed))
    model.fit(X_train, y_train)
    return model

def register_model(workspace, model):
    model = Model.register(workspace=workspace, model_path="outputs/model.pickle", model_name="churn-model")
    print('Churn model has been registered')


def main():
    workspace = run.experiment.workspace
    seed = 2022

    print('Training churn Model')
    result = build_features()
    model = train_model(result)

    persist_model(model)
    register_model(workspace, model)

if __name__ == '__main__':
    main()