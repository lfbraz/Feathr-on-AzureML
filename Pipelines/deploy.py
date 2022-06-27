from azureml.core import Environment
from azureml.core.run import Run
from azureml.core.model import InferenceConfig, Model
from azureml.core.compute import AksCompute

run = Run.get_context()
ws = run.experiment.workspace

churn_env = Environment.get(ws, 'ChurnModel-Env')

inference_config = InferenceConfig(entry_script="../Pipelines/entry.py",
                                   environment=churn_env)

cluster_name = 'inference-lab-01'
aks_target = AksCompute(ws, cluster_name)

model = ws.models['churn-model']

service = Model.deploy(workspace=ws,
                       name = 'api-churn-model',
                       models = [model],
                       inference_config = inference_config,
                       deployment_target = aks_target,
                       overwrite=True)
service.wait_for_deployment(show_output = True)