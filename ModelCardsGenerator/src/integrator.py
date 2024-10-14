from Utils.utility import templateRender
import sys

def textProcessing(text):
    """
    Processa il testo scritto dall'utente nel file 
    ModelCardsGenerator/Data/_parts.md
    """
    variables = {}
    sections = [section for section in text.split('\n\n') if section.strip()]

    for section in sections:
        name, content = section.split(':', 1)
        name = name.strip().replace(' ', '_').lower()
        content = content.strip()
        variables[name] = content

    description = {"text": variables.get('description')}
    how_to_use = {"text": variables.get('how_to_use')}
    intended_usage = {"text": variables.get('intended_usage')}
    limitations = {"text": variables.get('limitations')}
    
    data = [description, how_to_use, intended_usage, limitations]
    
    return data

def assembleDocs(data):
    """
    Assembla le diverse parti da integrare nella Model Card
    attraverso i templates 
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
    Controlla se la Model Card è già stata integrata,
    in caso positivo elimina la precedente integrazione.
    """
    with open(path, 'r') as modelCard:
        lines = modelCard.readlines()
        
        index = None
        for i, line in enumerate(lines):
            if line.startswith("## Description"):
                index = i
                break

    if index is not None:
        with open(path, 'w') as modelCard:
            for line in lines[:index]:
                modelCard.write(line)
    
if __name__ == "__main__":
    try:
        print("\nININ\n")
        path = sys.argv[1]

        isModelCardAssembled(path)

        with open('ModelCardsGenerator/src/Utils/textfiller.md', 'r') as file:
            text = file.read()
            print("\nINFILE\n")

        data = textProcessing(text)
        assembled = assembleDocs(data)

        with open(path, 'a') as modelCard:
            modelCard.write(assembled)
    except Exception as e:
        print(e)