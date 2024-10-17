from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import json, os

PATH = "ModelCardsGenerator/src/Utils/Templates"

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
    root = os.path.abspath(os.path.join(os.path.join(os.path.join(
        os.path.dirname(__file__), '..'), '..'), '..'))
    ModelCards_directory = os.path.join(root, 'ModelCards')
    path = os.path.join(ModelCards_directory, fname)
    saveEnv(path)
    return path

def saveEnv(path):
    subpath = path.split("ModelCards")[1].lstrip("/").lstrip("\\")
    path = os.path.join("ModelCards", subpath)
    with open (f"{PATH}/_parts/env.bin", 'wb') as env:
        env.write(path.encode())

def getEnv():
    with open (f"{PATH}/_parts/env.bin", 'rb') as env:
        return env.read().decode()

def templateRender(template, data, cd = ""):
    environment = Environment(loader = FileSystemLoader(f"{PATH}{cd}"))
    template = environment.get_template(template)
    return template.render(data)

def isUsable(text):
    lines = text.splitlines()
    titles = ["Description:", "How to use:", "Intended usage:", "Limitations:"]

    for index, line in enumerate(lines):
        if index > 0:
            for title in titles:
                if line == title and lines[index - 1].strip() != "+++++":
                    return False
    return True