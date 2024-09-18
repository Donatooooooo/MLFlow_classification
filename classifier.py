from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from mlflow.models import infer_signature
import sys, json, mlflow, pandas as pd
from os import path

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
        precision_micro = precision_score(y_test, y_pred, average='micro')
        recall_micro = recall_score(y_test, y_pred, average='micro')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        f1_macro = f1_score(y_test, y_pred, average='macro')

        print('Accuracy:', accuracy)
        print('Precision (micro):', precision_micro)
        print('Recall (micro):', recall_micro)
        print('F1_micro score:', f1_micro)
        print('F1_macro score:', f1_macro)
        print('\n')
        
        mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
        mlflow.set_experiment("Random Forest")

        with mlflow.start_run():
            tag = "Random forest"
            mlflow.set_tag("Training Info", tag)
    
            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric('Precision', precision_micro)
            mlflow.log_metric('Recall', recall_micro)
            mlflow.log_metric('F1_micro score', f1_micro)
            mlflow.log_metric('F1_macro score', f1_macro)

            mlflow.log_params(best_params)
            
            modelInfo = mlflow.sklearn.log_model(
                sk_model = model,
                artifact_path = "Random_Forest_Model",
                signature = infer_signature(X_test, model.predict(X_test)),
                input_example = X_test,
                registered_model_name = tag,
            )

            loadedModel = mlflow.pyfunc.load_model(modelInfo.model_uri)
            predictions = loadedModel.predict(X_test)
            featureNames = self.dataset.getDataset().columns.tolist()
            result = pd.DataFrame(X_test, columns = featureNames)
            result["actual_class"] = y_test
            result["predicted_class"] = predictions
            result[:10]
            
            result.to_csv('Dataset/predictions.csv', index=False)
            mlflow.log_artifact('Dataset/predictions.csv', "predictions.csv")
        mlflow.end_run()

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

    def trainRandomForestClassifier(self):
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        self.learning(model, param_grid, "RandomForestClassifier")

    def run(self):
        self.trainRandomForestClassifier()
