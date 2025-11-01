from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor

import numpy as np
import matplotlib.pyplot as plt
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
        self.results_path = OUTPUT_PATH / "classification"
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

        metrics_filename = os.path.join(self.results_path, f"metrics_{self.satellite_name}.json")

        try:
            with open(metrics_filename, 'w') as f:
                json.dump(self.results, f, indent=4)
            print("Metrics saved successfully.")
        except Exception as e:
            print(f"Error saving metrics as JSON: {e}")

        return self.results


class RegressionExperiment(Experiment):
    def __init__(self, X, y, satellite, train_size=0.8, test_size=0.1, val_size=0.1, split_type='train-val-test', print_stats=None):
        super().__init__(X, y, train_size, test_size, val_size, split_type, print_stats)
        self.satellite = satellite
        self.results_path = OUTPUT_PATH / "ml_experiment"

        self.__scale_data()

    
    def __scale_data(self):
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()

        self.X_train_scaled = self.x_scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.x_scaler.transform(self.X_test)

        self.y_train_scaled = self.y_scaler.fit_transform(self.y_train)
        self.y_test_scaled = self.y_scaler.transform(self.y_test)

        self.y_train = self.y_train.reshape(-1, )
        self.y_train_scaled = self.y_train_scaled.reshape(-1, )
        self.y_test = self.y_test.reshape(-1, )
        self.y_test_scaled = self.y_test_scaled.reshape(-1, )


    def fit_grid_search(self, model, param_grid, scaled=False, model_name="Model"):
        print(f"=== Running {model_name} for {self.satellite} ===")
        
        y_train = self.y_train
        y_test = self.y_test

        if scaled:
            X_train_data = self.X_train_scaled
            X_test_data = self.X_test_scaled
            
        else:
            X_train_data = self.X_train
            X_test_data = self.X_test

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train_data, y_train)

        best_model = grid_search.best_estimator_
        test_score = best_model.score(X_test_data, y_test)
        print(f"{model_name} Test R2 Score - {test_score*100:.4f}")
        
        y_preds = best_model.predict(X_test_data)

        self.make_plot(y_test, y_preds, model_name)
        return self.make_result_dict(y_test, y_preds)
    
    def make_result_dict(self, y_true, y_preds):
        result_dict = {}

        result_dict['MAE'] = round(mean_absolute_error(y_true, y_preds), 4)
        result_dict['MSE'] = round(mean_squared_error(y_true, y_preds), 4)
        result_dict['RMSE'] = round(root_mean_squared_error(y_true, y_preds), 4)
        result_dict['R2'] =  round(r2_score(y_true, y_preds), 4)
        result_dict['MAPE'] = round(mean_absolute_percentage_error(y_true, y_preds), 4)

        return result_dict

    def make_plot(self, y_test, y_preds, model_name):
        plot_dir = self.results_path / "plots"
        os.makedirs(plot_dir, exist_ok=True)

        mae = mean_absolute_error(y_test, y_preds)
        mape = mean_absolute_percentage_error(y_test, y_preds) * 100
        
        # Generate scatter plot with metrics
        plt.figure(figsize=(14, 7))
        indices = range(len(y_test))
        
        plt.scatter(indices, y_test, label='Actual', alpha=0.7, color='blue')
        plt.scatter(indices, y_preds, label='Predicted', alpha=0.7, color='red')
        
        plt.xlabel('Sample Index')
        plt.ylabel('Values')
        plt.title(f'{self.satellite}: Actual vs Predicted Values - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add text box with metrics
        metrics_text = f'MAE: {mae:.4f}\nMAPE: {mape:.2f}%'
        plt.annotate(metrics_text, xy=(0.02, 0.98), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    verticalalignment='top', fontsize=12)
        
        # Save plot
        plot_path = os.path.join(plot_dir, f"{self.satellite}_{model_name}_actual_vs_predicted.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

    def run_experiment(self):
        results = {}

        # Random Forest
        rf = RandomForestRegressor(random_state=10)
        rf_param_grid = {
            'n_estimators': [100, 200, 500],     
            'max_depth': [None, 5, 10, 20],      
            'min_samples_split': [2, 5, 10],     
            'min_samples_leaf': [1, 2, 4],       
            'max_features': ['sqrt', 'log2']     
        }
        results["RandomForest"] = self.fit_grid_search(rf, rf_param_grid, model_name="RandomForest")

        # XGBoost
        xgb = XGBRegressor(random_state=10, objective='reg:squarederror')
        xgb_param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        results["XGBoost"] = self.fit_grid_search(xgb, xgb_param_grid, model_name="XGBoost")

        # AdaBoost
        ada = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(random_state=10),
            random_state=10
        )
        ada_param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 1.0],
            'estimator__max_depth': [2, 3, 5, None],
            'estimator__min_samples_split': [2, 5, 10]
        }
        results["AdaBoost"] = self.fit_grid_search(ada, ada_param_grid, model_name="AdaBoost")

        # SVR (requires scaled data!)
        svr = SVR()
        svr_param_grid = {
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'epsilon': [0.01, 0.1, 0.2, 0.5]
        }
        if self.X_train_scaled is not None:  # only run if scaled data provided
            results["SVR"] = self.fit_grid_search(svr, svr_param_grid, scaled=True, model_name="SVR")
        
        metrics_filename = self.results_path / f"metrics_{self.satellite}.json"
        try:
            with open(metrics_filename, 'w') as f:
                json.dump(results, f, indent=4)
            print("Metrics saved successfully.")
        except Exception as e:
            print(f"Error saving metrics as JSON: {e}")
        
        return results
