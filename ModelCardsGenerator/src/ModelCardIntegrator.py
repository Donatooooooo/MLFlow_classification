from Utils.exceptions import TextValidationError
from Utils.utility import isUsable, getEnv
from Utils.utility import templateRender
from Utils.logger import Logger

def textProcessing(text):
    """
    Processes the text written by the user in the file 
    ModelCardsGenerator/Data/_parts.md
    """
    
    if not isUsable(text):
        raise TextValidationError()
    
    variables = {}
    sections = [section for section in text.split('+++++') if section.strip()]
    
    try:
        for section in sections:
            name, content = section.split(':', 1)
            name = name.strip().replace(' ', '_').lower()
            content = content.strip()
            variables[name] = content
    except ValueError:
        raise TextValidationError()

    description = {"text": variables.get('description')}
    how_to_use = {"text": variables.get('how_to_use')}
    intended_usage = {"text": variables.get('intended_usage')}
    limitations = {"text": variables.get('limitations')}
    
    data = [description, how_to_use, intended_usage, limitations]
    
    return data

def assembleDocs(data):
    """
    Assembles the various components to be integrated into the Model Card 
    using the templates.
    """
    templates = ["description_template.md", "how_to_use_template.md", 
                 "intended_usage_template.md", "limitations_template.md"]

    assembled = ""
    for i, template in enumerate(templates):
        if data[i].get("text"):
            instance = templateRender(template, data[i], "/_parts")
            assembled += f"{instance}\n"

    return assembled

def isModelCardAssembled(path):
    """
    Checks if the Model Card has already been integrated; 
    if so, removes the previous integration.
    """
    with open(path, 'r') as modelCard:
        lines = modelCard.readlines()
        
        index = None
        for i, line in enumerate(lines):
            if line.startswith(("## Description", "## Limitations", "## How to use", "## Intended usage")):
                index = i
                break

    if index is not None:
        with open(path, 'w') as modelCard:
            for line in lines[:index]:
                modelCard.write(line)
    
if __name__ == "__main__":
    output = Logger()
    try:
        path = getEnv()
        with open('ModelCardsGenerator/Data/add_info.md', 'r') as file:
            text = file.read()
        data = textProcessing(text)

        isModelCardAssembled(path)
        assembled = assembleDocs(data)
        with open(path, 'a') as modelCard:
            modelCard.write(assembled)

    except TextValidationError as e:
        output.log(f"Check add_info.md: {str(e)}")
    except FileNotFoundError as e:
        output.log(f"Check file path, {str(e).split('] ')[1]}")
    except Exception as e:
        output.log(f"Exception caused by: {str(e)}")
    finally:
        output.display()