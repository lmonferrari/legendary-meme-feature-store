import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from pandas import DataFrame


class Model:
    def __init__(
        self,
        X,
        y,
        test_size=0.2,
        model_params={"n_estimators": 100, "random_state": 42},
    ) -> None:
        self.X = X
        self.y = y
        self.test_size = test_size
        self.model_params = model_params

    def train_test_data_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size
        )

    def train(self):
        self.train_test_data_split()

        self.model = RandomForestClassifier(**self.model_params)
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        self.predicts = self.model.predict(self.X_test)
        score = accuracy_score(self.y_test, self.predicts)
        cls_report = classification_report(self.y_test, self.predicts)

        return score, cls_report

    def save_predicts(self, prediction_path):
        predicts_df = DataFrame({"Y_True": self.y_test, "Predictions": self.predicts})
        predicts_df.to_csv(prediction_path, index=False, sep=";")

    def save_model(self, model_path):
        joblib.dump(self.model, model_path)

    def save_info(self, run_info, run_info_path):
        with open(run_info_path, "w") as f:
            json.dump(run_info, f, indent=4)
