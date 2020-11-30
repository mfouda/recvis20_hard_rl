# tutorial/04-run-pytorch.py
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig

arguments = ['--horizon', 75, '--seed', 0, '--epochs', 2, '-cpu']

if __name__ == "__main__":
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name='day1-experiment-train')
    config = ScriptRunConfig(source_directory='nmp', script='train.py', compute_target='cpu-cluster', arguments=arguments)

    # set up pytorch environment
    env = Environment.from_conda_specification(name='nmprepr', file_path='.azureml/environment.yml')
    config.run_config.environment = env

    run = experiment.submit(config)

    aml_url = run.get_portal_url()
    print(aml_url)