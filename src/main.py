# from Dataset.dataset import Dataset
# from Models.randomForest import RandomForestTrainer
# from Models.knn import KNNTrainer
# from MLFlow import trainAndLog, ModelCard
# from Utils.utility import preprocessing
# import copy


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
from MLFlow import ModelCard

if __name__ == "__main__":
    input = sys.argv[1]
    parts = input.rsplit(' ', 1)
    
    print("MAINTEST:", parts[0], parts[1])
    ModelCard(parts[0], int(parts[1]))