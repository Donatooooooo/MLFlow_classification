from datetime import datetime
import json, os

def convertTime(unixTime):
    return datetime.fromtimestamp(unixTime/1000.0).strftime('%H:%M:%S %Y-%m-%d')

def extractInfoTags(tags):   
    data_tags = json.loads(tags.get('mlflow.log-model.history', ''))
    flavors = data_tags[0]['flavors']
    py_version = flavors['python_function']['python_version']
    lib = str([key for key in flavors.keys() if key != 'python_function'][0])
    lib_version = flavors[lib].get(f'{lib}_version')
    return py_version, lib, lib_version

def extratDatasetName(data):
    dataString = str(data)
    start = dataString.find("name='") + len("name='")
    end = dataString.find("'", start)
    return dataString[start:end]

def getPath(data):
    part = data.get("modelName").replace(" ", "")
    fname = f"{part}_v{data.get('version')}.md"
    root = os.path.abspath(os.path.join(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'), '..'))
    ModelCards_directory = os.path.join(root, 'ModelCards') 
    return os.path.join(ModelCards_directory, fname)