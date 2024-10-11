from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from os import path
import sys, json

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

class RandomForestTrainer:
    def __init__(self, target_column, drop_columns, dataset):
        self.dataset = dataset
        self.target_column = target_column
        self.drop_columns = drop_columns
        self.X, self.Y = self.loadData(dataset)

    def loadData(self, dataset):
        target = dataset.getColumn(self.target_column)
        dataset.dropDatasetColumns(self.drop_columns)
        X = dataset.getDataset()
        return X, target

    def evaluateModel(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        
        self.metrics ={
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': float(precision_score(y_test, y_pred, average='micro')),
            'Recall': float(recall_score(y_test, y_pred, average='micro')),
            'F1_micro score': float(f1_score(y_test, y_pred, average='micro')),
            'F1_macro score': float(f1_score(y_test, y_pred, average='macro'))
        }
        
        self.X = X_test
        self.Y = y_test
        
        print("Model trained and evaluated\n")

    def run(self):
        self.model.set_params(**self.params)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.evaluateModel(self.model, X_test, y_test)

    def findBestParams(self):
        name = "RandomForestClassifier"
        self.model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        best_params = self.loadBestParams(name)
        if best_params:
            print(f'Using saved best parameters for {name}:', best_params)
            self.params = best_params
            return None

        scorer = make_scorer(accuracy_score)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(self.model, param_grid, scoring=scorer, cv=cv, verbose=1)
        grid_search.fit(self.X, self.Y)

        best_params = grid_search.best_params_
        print(f'Best parameters for {name}:', best_params)
        
        self.saveBestParams(best_params, name)
        self.params = best_params
        return None
    
    def saveBestParams(self, best_params, name):
        with open('ModelTracker/Utils/best_params/best_params_' + name + '.json', 'w') as file:
            json.dump(best_params, file)

    def loadBestParams(self, name):
        filepath = 'ModelTracker//best_params/best_params_' + name + '.json'
        if path.exists(filepath):
            with open(filepath, 'r') as file:
                return json.load(file)
        return None
        
    def getMetrics(self):
        return self.metrics
    
    def getParams(self):
        return self.params
    
    def getModel(self):
        return self.model
    
    def getX(self):
        return self.X
    
    def getY(self):
        return self.Y
