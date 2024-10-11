from Dataset.dataset import Dataset
from Models.randomForest import RandomForestTrainer
from Models.knn import KNNTrainer
from MLFlow import trainAndLog, ModelCard
from Utils.utility import preprocessing
import copy


if False:
    dataset = Dataset("src/Dataset/brest_cancer.csv")
    dataset = preprocessing(dataset)

    experiment = "MultiClassifiers"

    RFdataset = copy.deepcopy(dataset)
    RFtrainer = RandomForestTrainer('diagnosis', ['diagnosis'], RFdataset)
    trainAndLog(
        dataset = dataset,
        trainer = RFtrainer,
        experimentName = experiment,
        datasetName = "brest_cancer.csv",
        modelName = "Random Forest Classifier",
        tags = {"Training Info": "testing with kMeans"}
    )

    KNNdataset = copy.deepcopy(dataset)
    KNNtrainer = KNNTrainer('diagnosis', ['diagnosis'], KNNdataset)
    trainAndLog(
        dataset = dataset,
        trainer = KNNtrainer,
        experimentName = experiment,
        datasetName = "brest_cancer.csv",
        modelName = "KNN Classifier",
        tags = {"Training Info": "testing with kMeans"}
    )

if False:
    ModelCard("KNN Classifier", 1)
    ModelCard("KNN Classifier", 3)
    ModelCard("Random forest with kMeans", 15)
    ModelCard("Random Forest Classifier", 3)

import sys

def process_string(input_string):
    result = input_string.upper()  # Qui semplicemente converto la stringa in maiuscolo
    return result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso corretto: python main.py <stringa>")
        sys.exit(1)

    input_string = sys.argv[1]
    result = process_string(input_string)
    
    print(f"Risultato: {result}")