from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import numpy as np
import pandas as pd
import json
import os

from constants import OUTPUT_PATH

class Experiment:
    def __init__(self, X, y, train_size=0.8, test_size=0.1, val_size=0.1, split_type='train-val-test', print_stats=None):
        self.X = X
        self.y = y
        
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        self.__split_data(train_size, test_size, val_size, split_type)
        if print_stats:
            self.__print_data_summary(train_size, test_size, val_size, split_type)
    
    def __split_data(self, train_size, test_size, val_size, split_type):
        if split_type == 'train-val-test':
            if not np.isclose(train_size + test_size + val_size, 1.0):
                raise ValueError("train_size, test_size, and val_size must sum to 1.0")
            
            if train_size == 1.0:
                self.X_train, self.y_train = self.X, self.y
                return

            X_train, X_temp, y_train, y_temp = train_test_split(
                self.X, self.y, train_size=train_size, random_state=42
            )
            
            self.X_train, self.y_train = X_train, y_train
            
            remaining_size = val_size + test_size
            if np.isclose(remaining_size, 0.0):
                return 

            relative_test_size = test_size / remaining_size
            
            if np.isclose(relative_test_size, 1.0):
                self.X_test, self.y_test = X_temp, y_temp
            elif np.isclose(relative_test_size, 0.0):
                self.X_val, self.y_val = X_temp, y_temp
            else:
                self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
                    X_temp, y_temp, test_size=relative_test_size, random_state=42
                )

        elif split_type == 'train-test':
            if not np.isclose(train_size + test_size, 1.0):
                raise ValueError("train_size and test_size must sum to 1.0")
            
            if train_size == 1.0:
                self.X_train, self.y_train = self.X, self.y
            elif test_size == 1.0:
                self.X_test, self.y_test = self.X, self.y
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    self.X, self.y, train_size=train_size, random_state=42
                )
        
        else:
            raise ValueError(f"Unknown split_type: {split_type}. Must be 'train-val-test' or 'train-test'.")
    
    def __print_data_summary(self, train_size, test_size, val_size, split_type):
        total_samples = len(self.X)
        train_samples = len(self.X_train) if self.X_train is not None else 0
        val_samples = len(self.X_val) if self.X_val is not None else 0
        test_samples = len(self.X_test) if self.X_test is not None else 0

        print(f"\n--- Experiment Data Initialized ---")
        print(f"Total Samples: {total_samples}")
        print(f"Split Type:    '{split_type}'")
        
        if split_type == 'train-val-test':
            print(f"  - Train: {train_size*100:>6.1f}% ({train_samples} samples)")
            print(f"  - Val:   {val_size*100:>6.1f}% ({val_samples} samples)")
            print(f"  - Test:  {test_size*100:>6.1f}% ({test_samples} samples)")
        elif split_type == 'train-test':
            print(f"  - Train: {train_size*100:>6.1f}% ({train_samples} samples)")
            print(f"  - Test:  {test_size*100:>6.1f}% ({test_samples} samples)")
        
        print(f"Total in splits: {train_samples + val_samples + test_samples}")
        print("-----------------------------------\n")


class ClassificationExperiment(Experiment):
    def __init__(self, X, y, satellite_name, labels, train_size=0.8, test_size=0.1, val_size=0.1, 
                 split_type='train-val-test', print_stats=True):
        super().__init__(X, y, train_size, test_size, val_size, split_type, print_stats)
        self.satellite_name = satellite_name
        self.results = {}
        self.results_path = OUTPUT_PATH
        self.labels = labels

        self.__scale_data()

    def __scale_data(self):
        self.ordinal_encoder = OrdinalEncoder(categories=[self.labels])
        self.target_encoder = TargetEncoder(categories=self.labels)


        self.y_train = self.ordinal_encoder.fit_transform(self.y_train)
        self.y_test = self.ordinal_encoder.transform(self.y_test)

        self.y_train = self.y_train.reshape(-1, )
        self.y_test = self.y_test.reshape(-1, )

    def run_experiment(self):
        if self.X_train is None or self.y_train is None:
            print("Error: Training data is not available. Cannot run experiment.")
            return

        models_to_run = {
            'rf': RandomForestClassifier(random_state=42),
            'xgb': XGBClassifier(random_state=42, eval_metric='mlogloss'),
            'ada': AdaBoostClassifier(random_state=42),
            'svc': SVC(probability=True, random_state=42)
        }
        
        os.makedirs(self.results_path, exist_ok=True)

        for name, model in models_to_run.items():
            print(f"\n--- Running Model: {name.upper()} ---")
            
            model.fit(self.X_train, self.y_train)

            model_results = {}

            if self.X_test is not None:
                
                test_preds = model.predict(self.X_test)
                test_accuracy = accuracy_score(self.y_test, test_preds)

                # Get dict report for clean JSON saving
                report_dict = classification_report(self.y_test, test_preds, zero_division=0, output_dict=True, target_names=self.labels)
                
                model_results['test_accuracy'] = test_accuracy
                model_results['test_classification_report'] = report_dict
                
            self.results[name] = model_results['test_classification_report']

            print(f"Test Acc - {model_results['test_accuracy']*100:.4f}%")

        metrics_filename = os.path.join(self.results_path, f"classification_metrics_{self.satellite_name}.json")
        
        try:
            with open(metrics_filename, 'w') as f:
                json.dump(self.results, f, indent=4)
            print("Metrics saved successfully.")
        except Exception as e:
            print(f"Error saving metrics as JSON: {e}")

        return self.results
