from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor

from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import tensorflow as tf

from mapie.utils import train_conformalize_test_split
from mapie.regression import ConformalizedQuantileRegressor
from mapie.metrics.regression import regression_coverage_score, regression_mean_width_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

from constants import OUTPUT_PATH

from tqdm import tqdm

class EpochTqdm(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, desc="Epochs"):
        super().__init__()
        self.pbar = tqdm(total=total_epochs, desc=desc, unit="epoch")
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # show loss/val_loss nicely (change keys as needed)
        postfix = {}
        if "loss" in logs: postfix["loss"] = f"{logs['loss']:.4f}"
        if "val_loss" in logs: postfix["val_loss"] = f"{logs['val_loss']:.4f}"
        if postfix:
            self.pbar.set_postfix(postfix)
        self.pbar.update(1)
    def on_train_end(self, logs=None):
        self.pbar.close()



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
                 split_type='train-val-test', print_stats=True, type='censored'):
        super().__init__(X, y, train_size, test_size, val_size, split_type, print_stats)
        self.satellite_name = satellite_name
        self.results = {}
        self.results_path = OUTPUT_PATH / f"classification_{type}"
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
    def __init__(self, X, y, satellite, train_size=0.8, test_size=0.1, val_size=0.1, 
                 split_type='train-val-test', print_stats=None, type='censored'):
        super().__init__(X, y, train_size, test_size, val_size, split_type, print_stats)
        self.satellite = satellite
        self.results_path = OUTPUT_PATH / f"ml_experiment_{type}"

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


class PredictionIntervalEstimation(Experiment):
    def __init__(self, X, y, satellite, train_size=0.8, test_size=0.1, val_size=0.1, split_type='train-val-test', print_stats=None):
        super().__init__(X, y, train_size, test_size, val_size, split_type, print_stats)
        self.results_path = OUTPUT_PATH / "pi_estimation_censored"
        self.satellite = satellite

        self.__scale_data()

    def __scale_data(self):
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()

        self.X_train_scaled = self.x_scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.x_scaler.transform(self.X_test)
        self.X_val_scaled = self.x_scaler.transform(self.X_val)

        self.y_train_scaled = self.y_scaler.fit_transform(self.y_train)
        self.y_test_scaled = self.y_scaler.transform(self.y_test)
        self.y_val_scaled = self.y_scaler.transform(self.y_val)

        self.y_train = self.y_train.reshape(-1, )
        self.y_train_scaled = self.y_train_scaled.reshape(-1, )
        self.y_test = self.y_test.reshape(-1, )
        self.y_test_scaled = self.y_test_scaled.reshape(-1, )
        self.y_val = self.y_val.reshape(-1, )
        self.y_val_scaled = self.y_val_scaled.reshape(-1, )

    def pinball_loss(self, y_true, y_pred, tau):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(tau * error, (tau - 1) * error))

    def lower_quantile_loss(self, y_true, y_pred):
        return self.pinball_loss(y_true, y_pred, tau=0.025)
    
    def upper_quantile_loss(self, y_true, y_pred):
        return self.pinball_loss(y_true, y_pred, tau=0.975)

    def train_model(self, model, learning_rate, optimizer='adam', epochs=200, batch_size=32, verbose=0):
        self.upper_model = tf.keras.models.clone_model(model)
        self.lower_model = tf.keras.models.clone_model(model)

        if isinstance(optimizer, str):
            optimizer_config = {'class_name': optimizer, 'config': {'learning_rate': learning_rate}} # Adam default
        else:
            optimizer_config = optimizer.get_config()
            optimizer_config['class_name'] = optimizer_config['name']
            del optimizer_config['name']

        # 2. Create two new, independent optimizer instances from that config
        upper_optimizer = tf.keras.optimizers.get(optimizer_config.copy())
        lower_optimizer = tf.keras.optimizers.get(optimizer_config.copy())
        # compile both models
        self.lower_model.compile(
            optimizer=lower_optimizer,
            loss=self.lower_quantile_loss
        )
        self.upper_model.compile(
            optimizer=upper_optimizer,
            loss=self.upper_quantile_loss
        )


        # print("--------- TRAINING UPPER MODEL -----------\n")
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        progress = EpochTqdm(total_epochs=epochs)
        self.upper_model_history = self.upper_model.fit(
            self.X_train_scaled, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val_scaled, self.y_val),
            verbose=verbose,
            callbacks=[progress, early_stopping]
        )
        # print("--------- TRAINING LOWER MODEL -----------\n")
        progress = EpochTqdm(total_epochs=epochs)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.lower_model_history = self.lower_model.fit(
            self.X_train_scaled, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val_scaled, self.y_val),
            verbose=verbose,
            callbacks=[progress, early_stopping]
        )

        # Predict on both test and validation sets
        y_preds_lower_test = self.lower_model.predict(self.X_test_scaled)
        y_preds_upper_test = self.upper_model.predict(self.X_test_scaled)
        y_preds_lower_val = self.lower_model.predict(self.X_val_scaled)
        y_preds_upper_val = self.upper_model.predict(self.X_val_scaled)

        return (y_preds_lower_test.flatten(), y_preds_upper_test.flatten(),
                y_preds_lower_val.flatten(), y_preds_upper_val.flatten())

    def plot_training_history(self, model_param_string):
        plt.figure(figsize=(12, 4))
        plt.suptitle(f"{self.satellite}: {model_param_string}\nUpper and Lower Model Training Loss")
        plt.subplot(1, 2, 1)
        plt.plot(self.upper_model_history.history['loss'], label='Training Loss')
        plt.plot(self.upper_model_history.history['val_loss'], label='Validation Loss')
        plt.title('Upper Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.lower_model_history.history['loss'], label='Training Loss')
        plt.plot(self.lower_model_history.history['val_loss'], label='Validation Loss')
        plt.title('Lower Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def evaluate_model(self, y_true, y_pred_lower, y_pred_upper):
        
        def picp(y_true_vals, y_pred_lower_vals, y_pred_upper_vals):
            """Prediction Interval Coverage Probability"""
            covered = np.sum((y_true_vals >= y_pred_lower_vals) & (y_true_vals <= y_pred_upper_vals))
            return covered / len(y_true_vals)

        def mpiw(y_pred_lower_vals, y_pred_upper_vals):
            """Mean Prediction Interval Width"""
            return np.mean(y_pred_upper_vals - y_pred_lower_vals)

        return {
            'PICP': float(picp(y_true, y_pred_lower, y_pred_upper)),
            'MPIW': float(mpiw(y_pred_lower, y_pred_upper))
        }

    def plot_prediction_interval(self, y_pred_lower_test, y_pred_upper_test, y_pred_lower_val, y_pred_upper_val, model_param_string):
        indices = range(len(self.y_test))

        # Calculate metrics for both test and validation sets
        test_eval_dict = self.evaluate_model(self.y_test, y_pred_lower_test, y_pred_upper_test)
        val_eval_dict = self.evaluate_model(self.y_val, y_pred_lower_val, y_pred_upper_val)

        plt.figure(figsize=(14, 7))
        plt.plot(indices, self.y_test, 'o', color='blue', label='Actual Soil Moisture (Test Set)')
        plt.plot(indices, y_pred_lower_test, color='red', linestyle='--', label='Lower Bound')
        plt.plot(indices, y_pred_upper_test, color='orange', linestyle='--', label='Upper Bound')

        plt.fill_between(indices, y_pred_lower_test, y_pred_upper_test, color='gray', alpha=0.2, label='95% Prediction Interval')

        # Create a clean, aligned text box for both metrics
        test_metrics = f"Test  | PICP: {test_eval_dict['PICP']*100:5.2f}% | MPIW: {test_eval_dict['MPIW']:.4f}"
        val_metrics =  f"Valid | PICP: {val_eval_dict['PICP']*100:5.2f}% | MPIW: {val_eval_dict['MPIW']:.4f}"
        metrics_text = f"{test_metrics}\n{val_metrics}"
        
        plt.annotate(metrics_text, xy=(0.02, 0.98), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8), 
                    verticalalignment='top', fontsize=12,
                    # Using a monospaced font for clean alignment
                    fontname='monospace')
        
        plot_dir = self.results_path / "plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.xlabel('Sample Index')
        plt.ylabel('Soil Moisture')
        plt.title(f'{self.satellite}: {model_param_string}\nPrediction Interval for Soil Moisture')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{plot_dir}/{self.satellite}_{model_param_string}.png", dpi=300)
        # plt.show()
        plt.close()

    def run_experiment(self, model, optimizer='adam', epochs=200, learning_rate=0.01, batch_size=32, verbose=0, model_param_string=None):
        # Unpack all four returned prediction arrays
        y_preds_lower_test, y_preds_upper_test, y_preds_lower_val, y_preds_upper_val = self.train_model(
            model, optimizer=optimizer, epochs=epochs, batch_size=batch_size, verbose=verbose, learning_rate=learning_rate
        )
        
        # self.plot_training_history(model_param_string)
        
        # Pass all four arrays to the plotting function
        self.plot_prediction_interval(
            y_preds_lower_test, y_preds_upper_test,
            y_preds_lower_val, y_preds_upper_val,
            model_param_string
        )

        results_val = self.evaluate_model(self.y_val, y_preds_lower_val, y_preds_upper_val)
        results_test = self.evaluate_model(self.y_test, y_preds_lower_test, y_preds_upper_test)

        results = {
            "val": results_val,
            "test": results_test
        }
        print(f"{model_param_string}: {json.dumps(results, indent=4)}")
        # with open(self.results_path / f"{self.satellite}_metrics.json", "w") as f:
        #     json.dump(results, f, indent=4)

        return results


class ConformalRegression:
    def __init__(self, X, y, satellite, train_size=0.8, conf_size=0.1, test_size=0.1, print_splits=True, type='uncensored'):
        self.X = X
        self.y = y
        self.satellite = satellite
        self.train_size = train_size
        self.conf_size = conf_size
        self.test_size = test_size
        self.random_seed = 42
        self.results_path = OUTPUT_PATH / f"conformal_regression_{type}"
        self.__prepare_data(print_splits)
    
    def __prepare_data(self, print_splits):
        self.y = self.y.reshape(-1, )

        (
            X_train, X_cal, X_test,
            y_train, y_cal, y_test
        ) = train_conformalize_test_split(self.X, self.y, 
                                          train_size=self.train_size, conformalize_size=self.conf_size, 
                                          test_size=self.test_size, random_state=self.random_seed)

        self.X_train = X_train
        self.X_conf = X_cal
        self.X_test = X_test

        self.y_train = y_train.reshape(-1, )
        self.y_conf = y_cal.reshape(-1, )
        self.y_test = y_test.reshape(-1, )

        if print_splits:
            total = len(self.X)
            print(
                "Split sizes:\n"
                f"  Train    - X: {getattr(self.X_train, 'shape', (len(self.X_train),))}, y: {len(self.y_train)} ({len(self.y_train)/total:.1%})\n"
                f"  Conformal- X: {getattr(self.X_conf,  'shape', (len(self.X_conf), ))}, y: {len(self.y_conf)} ({len(self.y_conf)/total:.1%})\n"
                f"  Test     - X: {getattr(self.X_test,  'shape', (len(self.X_test), ))}, y: {len(self.y_test)} ({len(self.y_test)/total:.1%})\n"
                f"  Original data rows: {total}"
            )

    def run_experiment(self):
        models = {
            "QuantileRegressor": QuantileRegressor(),
            "GradientBoostingRegressor": GradientBoostingRegressor(loss="quantile"),
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor(loss="quantile"),
            # "LGBMRegressor": LGBMRegressor(objective="quantile")
        }

        results = {}

        for model_string, model in models.items():
            print(f"==========Running {model_string}===========\n\n")
            regressor = ConformalizedQuantileRegressor(
                estimator=model,
                confidence_level=0.95,
                prefit=False
            )
            regressor.fit(self.X_train, self.y_train)
            regressor.conformalize(self.X_conf, self.y_conf)

            _, intervals = regressor.predict_interval(self.X_test, minimize_interval_width=True)

            intervals = intervals.squeeze()
            lower_preds = intervals[:, 0]
            upper_preds = intervals[:, 1]
            self.plot_prediction_interval(lower_preds, upper_preds, model_string)
            results[model_string] = self.evaluate_model(self.y_test, lower_preds, upper_preds)
        
        with open(self.results_path / f"{self.satellite}_metrics.json", "w") as f:
            json.dump(results, f, indent=4)
    
    def evaluate_model(self, y_true, y_pred_lower, y_pred_upper):
        
        def picp(y_true_vals, y_pred_lower_vals, y_pred_upper_vals):
            """Prediction Interval Coverage Probability"""
            covered = np.sum((y_true_vals >= y_pred_lower_vals) & (y_true_vals <= y_pred_upper_vals))
            return covered / len(y_true_vals)

        def mpiw(y_pred_lower_vals, y_pred_upper_vals):
            """Mean Prediction Interval Width"""
            return np.mean(y_pred_upper_vals - y_pred_lower_vals)

        return {
            'PICP': float(picp(y_true, y_pred_lower, y_pred_upper)),
            'MPIW': float(mpiw(y_pred_lower, y_pred_upper))
        }

    def plot_prediction_interval(self, y_pred_lower_test, y_pred_upper_test, model_param_string):
        indices = range(len(self.y_test))

        # Calculate metrics for both test and validation sets
        test_eval_dict = self.evaluate_model(self.y_test, y_pred_lower_test, y_pred_upper_test)

        plt.figure(figsize=(14, 7))
        plt.plot(indices, self.y_test, 'o', color='blue', label='Actual Soil Moisture (Test Set)')
        plt.plot(indices, y_pred_lower_test, color='red', linestyle='--', label='Lower Bound')
        plt.plot(indices, y_pred_upper_test, color='orange', linestyle='--', label='Upper Bound')

        plt.fill_between(indices, y_pred_lower_test, y_pred_upper_test, color='gray', alpha=0.2, label='95% Prediction Interval')

        # Create a clean, aligned text box for both metrics
        test_metrics = f"Test  | PICP: {test_eval_dict['PICP']*100:5.2f}% | MPIW: {test_eval_dict['MPIW']:.4f}"
        metrics_text = f"{test_metrics}"
        
        plt.annotate(metrics_text, xy=(0.02, 0.98), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8), 
                    verticalalignment='top', fontsize=12,
                    # Using a monospaced font for clean alignment
                    fontname='monospace')
        
        plot_dir = self.results_path / "plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.xlabel('Sample Index')
        plt.ylabel('Soil Moisture')
        plt.title(f'{self.satellite}: {model_param_string}\nPrediction Interval for Soil Moisture')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{plot_dir}/{self.satellite}_{model_param_string}.png", dpi=300)
        # plt.show()
        plt.close()