from Dataset.dataset import Dataset
from Classifiers.randomForest import RandomForestTrainer
from Classifiers.knn import KNNTrainer
from MLFlow import trainAndLog, createMD
from Utils.utility import preprocessing
import warnings, copy
warnings.filterwarnings("ignore")


dataset = Dataset("Dataset/brest_cancer.csv")
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

createMD("KNN Classifier", 1)
createMD("KNN Classifier", 3)
createMD("Random forest with kMeans", 15)
createMD("Random Forest Classifier", 3)