from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from os import path
import sys, json

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

class ModelTrainerClass:
    def __init__(self, target_column, drop_columns, dataset):
        self.dataset = dataset
        self.target_column = target_column
        self.drop_columns = drop_columns
        self.X, self.y = self.loadData(dataset)

    def loadData(self, dataset):
        target = dataset.getColumn(self.target_column)
        dataset.dropDatasetColumns(self.drop_columns)
        X = dataset.getDataset()
        return X, target

    def evaluateModel(self, model, X_test, y_test, best_params):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision_micro = float(precision_score(y_test, y_pred, average='micro'))
        recall_micro = float(recall_score(y_test, y_pred, average='micro'))
        f1_micro = float(f1_score(y_test, y_pred, average='micro'))
        f1_macro = float(f1_score(y_test, y_pred, average='macro'))

        self.metrics ={
            'Accuracy': accuracy,
            'Precision': precision_micro,
            'Recall': recall_micro,
            'F1_micro score': f1_micro,
            'F1_macro score': f1_macro
        }
        self.params = best_params
        self.model = model
        self.X = X_test
        self.Y = y_test
        
        print("Model trained and evaluated\n")

    def saveBestParams(self, best_params, name):
        with open('Evaluation/best_params_' + name + '.json', 'w') as file:
            json.dump(best_params, file)

    def loadBestParams(self, name):
        filepath = 'Evaluation/best_params_' + name + '.json'
        if path.exists(filepath):
            with open(filepath, 'r') as file:
                return json.load(file)
        return None

    def learning(self, model, param_grid, name):
        best_params = self.loadBestParams(name)
        if best_params:
            print(f'Using saved best parameters for {name}:', best_params)
            model.set_params(**best_params)
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            self.evaluateModel(model, X_test, y_test, best_params)
            return model

        scorer = make_scorer(accuracy_score)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        grid_search = GridSearchCV(model, param_grid, scoring=scorer, cv=cv, verbose=1)
        grid_search.fit(self.X, self.y)

        print(f'Best parameters for {name}:', grid_search.best_params_)
        print(f'Best cross-validation score for {name}:', grid_search.best_score_)

        self.saveBestParams(grid_search.best_params_, name)

        best_model = grid_search.best_estimator_

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        best_model.fit(X_train, y_train)
        self.evaluateModel(best_model, X_test, y_test, grid_search.best_params_)

        return best_model

    def run(self):
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        self.learning(model, param_grid, "RandomForestClassifier")
        
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
