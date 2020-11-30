# tutorial/01-create-workspace.py
from azureml.core import Workspace

ws = Workspace.create(name='recvis20', # provide a name for your workspace
                      subscription_id='2f834842-d913-4925-8d21-9e6816a22ba9', # provide your subscription ID
                      resource_group='recvis20_base_line', # provide a resource group name
                      create_resource_group=True,
                      location='westeurope') # For example: 'westeurope' or 'eastus2' or 'westus2' or 'southeastasia'.

# write out the workspace details to a configuration file: .azureml/config.json
ws.write_config(path='.azureml')