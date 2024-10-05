from Dataset.dataset import Dataset
from classifier import ModelTrainerClass
from MLFlow import trainAndLog, createMD
from util.util import preprocessing


dataset = Dataset("Dataset/brest_cancer.csv")
dataset = preprocessing(dataset)

trainer = ModelTrainerClass('diagnosis', ['diagnosis'], dataset)
trainAndLog(dataset, trainer, "RFclassifier v2", "brest_cancer.csv", "Random Forest")

createMD("Random Forest", 9)