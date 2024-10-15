from Dataset.dataset import Dataset
from Classifiers.randomForest import RandomForestTrainer
from Classifiers.knn import KNNTrainer
from MLFlowTracker import trainAndLog
from Utils.utility import preprocessing
import copy


dataset = Dataset("ModelTracker/Dataset/brest_cancer.csv")
dataset = preprocessing(dataset)

experiment = "MultiClassifiers"

RFdataset = copy.deepcopy(dataset)
RFtrainer = RandomForestTrainer('diagnosis', ['diagnosis'], RFdataset)
trainAndLog(
    dataset = dataset,
    trainer = RFtrainer,
    experimentName = experiment,
    datasetName = "brest_cancer.csv",
    modelName = "Random Forest",
    tags = {"Training Info": "testing with kMeans"}
)

KNNdataset = copy.deepcopy(dataset)
KNNtrainer = KNNTrainer('diagnosis', ['diagnosis'], KNNdataset)
trainAndLog(
    dataset = dataset,
    trainer = KNNtrainer,
    experimentName = experiment,
    datasetName = "brest_cancer.csv",
    modelName = "KNN",
    tags = {"Training Info": "testing with kMeans"}
)