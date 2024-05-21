import numpy as np
from decision_tree import DecisionTree
import time
import pandas as pd

class RandomForestClassifier():
    """
    Random Forest Classifier
    Training: Use "train" function with train set features and labels
    Predicting: Use "predict" function with test set features
    """

    def __init__(self, n_base_learner=10, max_depth=5, min_samples_leaf=1, min_information_gain=0.0, \
                 numb_of_features_splitting=None, bootstrap_sample_size=None) -> None:
        self.n_base_learner = n_base_learner
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.numb_of_features_splitting = numb_of_features_splitting
        self.bootstrap_sample_size = bootstrap_sample_size
        self.total_execution_times = {
            "_create_bootstrap_samples": 0,
            "train": 0,
            "_calculate_rf_feature_importance": 0,
            "predict": 0,
            "predict_proba": 0,
            "_predict_proba_w_base_learners": 0
        }


    def _create_bootstrap_samples(self, X, Y) -> tuple:
        """
        Creates bootstrap samples for each base learner
        """
        start_time = time.time()
        bootstrap_samples_X = []
        bootstrap_samples_Y = []

        for i in range(self.n_base_learner):
            
            if not self.bootstrap_sample_size:
                self.bootstrap_sample_size = X.shape[0]
            
            sampled_idx = np.random.choice(X.shape[0], size=self.bootstrap_sample_size, replace=True)
            bootstrap_samples_X.append(X[sampled_idx])
            bootstrap_samples_Y.append(Y[sampled_idx])
        end_time = time.time()

        execution_time = end_time - start_time
        self.total_execution_times["_create_bootstrap_samples"] += execution_time

        return bootstrap_samples_X, bootstrap_samples_Y

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        start_time = time.time()
        """Trains the model with given X and Y datasets"""
        bootstrap_samples_X, bootstrap_samples_Y = self._create_bootstrap_samples(X_train, Y_train)

        self.base_learner_list = []
        for base_learner_idx in range(self.n_base_learner):
            base_learner = DecisionTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, \
                                        min_information_gain=self.min_information_gain, 
                                        numb_of_features_splitting=self.numb_of_features_splitting)
            
            base_learner.train(bootstrap_samples_X[base_learner_idx], bootstrap_samples_Y[base_learner_idx])
            self.base_learner_list.append(base_learner)

        # Calculate feature importance
        self.feature_importances = self._calculate_rf_feature_importance(self.base_learner_list)
        end_time = time.time()
        execution_time = end_time - start_time
        self.total_execution_times["train"] += execution_time

    def _predict_proba_w_base_learners(self,  X_set: np.array) -> list:
        """
        Creates list of predictions for all base learners
        """
        start_time = time.time()
        pred_prob_list = []
        for base_learner in self.base_learner_list:
            pred_prob_list.append(base_learner.predict_proba(X_set))
        end_time = time.time()
        execution_time = end_time - start_time
        self.total_execution_times["_predict_proba_w_base_learners"] += execution_time
        return pred_prob_list

    def predict_proba(self, X_set: np.array) -> list:
        """Returns the predicted probs for a given data set"""
        start_time = time.time()
        pred_probs = []
        base_learners_pred_probs = self._predict_proba_w_base_learners(X_set)

        # Average the predicted probabilities of base learners
        for obs in range(X_set.shape[0]):
            base_learner_probs_for_obs = [a[obs] for a in base_learners_pred_probs]
            # Calculate the average for each index
            obs_average_pred_probs = np.mean(base_learner_probs_for_obs, axis=0)
            pred_probs.append(obs_average_pred_probs)
        end_time = time.time()
        execution_time = end_time - start_time
        self.total_execution_times["predict_proba"] += execution_time
        return pred_probs

    def predict(self, X_set: np.array) -> np.array:
        """Returns the predicted labels for a given data set"""
        start_time = time.time()
        pred_probs = self.predict_proba(X_set)
        preds = np.argmax(pred_probs, axis=1)
        end_time = time.time()
        execution_time = end_time - start_time
        self.total_execution_times["predict"] += execution_time
        return preds
    
    def _calculate_rf_feature_importance(self, base_learners):
        start_time = time.time()
        """Calcalates the average feature importance of the base learners"""
        feature_importance_dict_list = []
        for base_learner in base_learners:
            feature_importance_dict_list.append(base_learner.feature_importances)

        feature_importance_list = [list(x.values()) for x in feature_importance_dict_list]
        average_feature_importance = np.mean(feature_importance_list, axis=0)
        end_time = time.time()
        execution_time = end_time - start_time
        self.total_execution_times["_calculate_rf_feature_importance"] += execution_time
        return average_feature_importance
    
    def print_time(self) -> None:
        print("Tổng thời gian thực thi của các phương thức:")
        for method, execution_time in self.total_execution_times.items():
            print(f"{method}: {execution_time} seconds")

    def time_df(self):
        components = []

        for method, execution_time in self.total_execution_times.items():
            components.append({
                "Method": method,
                "Execution_Time": execution_time
            })

        return pd.DataFrame.from_records(components).sort_values(by=["Execution_Time"], ascending=False)