# {{ modelName }} - {{ version }}
## General Information 
- Developed by: {{ author }}
- Model Type: {{ modelType }}
- {{ library }} version: {{ libraryVersion }}
- Python version: {{ pythonVersion }}
## Training Details
- Dataset: {{ datasetName }}
- Parameters: 
    {% for param, val in parameters.items() -%}
    - `{{ param }}` {{ val }}
    {% endfor %}
- Training started at: {{ startTime }}
- Training ended at: {{ endTime }}
## Evaluation
{% for metric, val in evaluations.items() -%}
- `{{ metric }}` {{ val }}
{% endfor -%}