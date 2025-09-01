from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LassoLars, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import PredictionErrorDisplay
from xgboost import XGBRegressor

import tensorflow as tf

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

warnings.filterwarnings('ignore', category=FutureWarning)



class ModelExperimentsV1:
    def __init__(self, X, y, satellite, train_size=0.8):
        split_idx = int(len(X) * train_size)
        
        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx].values.flatten()
        self.y_test = y.iloc[split_idx:].values.flatten()
        
        # Store scaler for potential use, but consider moving scaling into pipeline
        self.scaler = MinMaxScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        self.satellite = satellite


    def fit_grid_search(self, model, param_grid, scaled=False, model_name="Model"):
        print(f"=== Running {model_name} for {self.satellite} ===")
        
        y_train = self.y_train
        y_test = self.y_test

        if scaled:
            pipeline = Pipeline([
                ('scaler', MinMaxScaler()),
                ('model', model)
            ])
            adjusted_param_grid = {f'model__{k}': v for k, v in param_grid.items()}
            
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=adjusted_param_grid,
                cv=3,
                n_jobs=-1,
                verbose=1
            )
            X_train_data = self.X_train
            X_test_data = self.X_test
            
        else:
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                verbose=1
            )
            X_train_data = self.X_train
            X_test_data = self.X_test

        grid_search.fit(X_train_data, y_train)
        # print("Best Parameters:", grid_search.best_params_)

        best_model = grid_search.best_estimator_
        test_score = best_model.score(X_test_data, y_test)
        # print("Test R2 Score:", test_score)
        
        y_preds = best_model.predict(X_test_data)

        mae = mean_absolute_error(y_test, y_preds)
        mape = mean_absolute_percentage_error(y_test, y_preds) * 100

        
        # Create output directory if it doesn't exist
        plot_dir = Path(f"/home/kshipra/work/major/ml experiments/output/plots/{self.satellite}")
        os.makedirs(plot_dir, exist_ok=True)

        
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
        plot_path = os.path.join(plot_dir, f"{model_name}_actual_vs_predicted.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return self.make_result_dict(y_test, y_preds)
    
    def make_result_dict(self, y_true, y_preds):
        result_dict = {}

        result_dict['MAE'] = round(mean_absolute_error(y_true, y_preds), 4)
        result_dict['MSE'] = round(mean_squared_error(y_true, y_preds), 4)
        result_dict['RMSE'] = round(root_mean_squared_error(y_true, y_preds), 4)
        result_dict['R2'] =  round(r2_score(y_true, y_preds), 4)
        result_dict['MAPE'] = round(mean_absolute_percentage_error(y_true, y_preds), 4)

        return result_dict

    def run_all(self):
        """
        Runs all experiments: RF, XGB, AdaBoost, SVR
        """
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
        
        lasso = Lasso()


        return results


class ANNExperimentsV1:
    def __init__(self, data, features=['HH', 'HV'], target='SM', test_size=0.1, val_size=0.25, random_state=42, satellite=None):
        self.data = data
        self.features = features
        self.target = target
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.satellite = satellite
        
        # Initialize attributes
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = None
        self.X_train_scaled = None
        self.X_val_scaled = None
        self.X_test_scaled = None
        self.model = None
        self.history = None
        
        # Prepare data
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare and split the data into train, validation, and test sets."""
        X = self.data[self.features]
        y = self.data[self.target]
        
        # Split the data into train, validation, and test sets
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=self.val_size, random_state=self.random_state
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Validation set: {self.X_val.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        # Scale the features
        self.scaler = MinMaxScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
    
    def train_model(self, model, optimizer='adam', epochs=200, batch_size=32, verbose=0):
        self.model = model
        
        # Compile the model
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        # Train the model
        self.history = self.model.fit(
            self.X_train_scaled, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val_scaled, self.y_val),
            verbose=verbose
        )
        
        # Evaluate on test set
        test_loss, test_mae = self.model.evaluate(self.X_test_scaled, self.y_test, verbose=0)
        print(f"\nTest Loss (MSE): {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test_scaled).flatten()
        
        # Calculate additional metrics
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        print(f"\nAdditional Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {np.sqrt(mse):.4f}")
        
        return y_pred, test_loss, test_mae, mse, r2
    
    def plot_training_history(self):
        """Plot training and validation loss and MAE."""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'], label='Training MAE')
        plt.plot(self.history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions_vs_actual(self, y_pred):
        """Plot predictions vs actual values."""
        
        plt.figure(figsize=(8, 6))

        plt.scatter(self.y_test, y_pred, alpha=0.6)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                 [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')

        plt.title('Actual vs Predicted Values')

        
        
        plt.show()
    
    def plot_line_comparison(self, y_pred, model_params):
        """Plot line comparison of actual vs predicted values."""
        plt.figure(figsize=(14, 7))

        mae = mean_absolute_error(self.y_test, y_pred)
        mape = mean_absolute_percentage_error(self.y_test, y_pred) * 100
        
        # Create index for x-axis
        indices = range(len(self.y_test))
        
        # Plot both lines
        plt.scatter(indices, self.y_test.values, label='actual', color='blue', alpha=0.7)
        plt.scatter(indices, y_pred, label='predicted', color='red', alpha=0.7)
        
        plt.xlabel('Sample Index')
        plt.ylabel('Values')
        plt.title(f'{self.satellite}: Actual vs Predicted Values.\n Params: {model_params}')

        metrics_text = f'MAE: {mae:.4f}\nMAPE: {mape:.2f}%'
        plt.annotate(metrics_text, xy=(0.02, 0.98), xycoords='axes fraction', 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), 
                     verticalalignment='top', fontsize=12)

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'/home/kshipra/work/major/ml experiments/output/plots/{self.satellite}/{model_params}.png')
        plt.show()
    
    def run_experiment(self, model, optimizer='adam', epochs=200, batch_size=32, verbose=0, model_param_string=None):
        # Train model
        y_pred, test_loss, test_mae, mse, r2 = self.train_model(
            model, optimizer, epochs, batch_size, verbose
        )
        
        # Generate plots
        self.plot_training_history()
        # self.plot_predictions_vs_actual(y_pred)
        self.plot_line_comparison(y_pred, model_param_string)


class PredictionIntervalEstimation(ANNExperimentsV1):
    def __init__(self, data, features=['HH', 'HV'], target='SM', test_size=0.1, val_size=0.25, random_state=42, satellite=None):
        super().__init__(data, features=['HH', 'HV'], target='SM', test_size=0.1, val_size=0.25, random_state=42, satellite=None)

    def pinball_loss(self, y_true, y_pred, tau):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(tau * error, (tau - 1) * error))

    def lower_quantile_loss(self, y_true, y_pred):
        return self.pinball_loss(y_true, y_pred, tau=0.025)
    
    def upper_quantile_loss(self, y_true, y_pred):
        return self.pinball_loss(y_true, y_pred, tau=0.975)

    def train_model(self, model, optimizer='adam', epochs=200, batch_size=32, verbose=0):
        self.upper_model = tf.keras.models.clone_model(model)
        self.lower_model = tf.keras.models.clone_model(model)

        # compile both models
        self.lower_model.compile(
            optimizer=optimizer,
            loss=self.lower_quantile_loss
        )
        self.upper_model.compile(
            optimizer=optimizer,
            loss=self.upper_quantile_loss
        )

        print("\n--------- TRAINING UPPER MODEL -----------\n")
        self.upper_model_history = self.upper_model.fit(
            self.X_train_scaled, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val_scaled, self.y_val),
            verbose=verbose
        )

        print("\n--------- TRAINING LOWER MODEL -----------\n")
        self.lower_model_history = self.lower_model.fit(
            self.X_train_scaled, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val_scaled, self.y_val),
            verbose=verbose
        )

        y_preds_lower = self.lower_model.predict(self.X_test_scaled)
        y_preds_upper = self.upper_model.predict(self.X_test_scaled)

        return y_preds_lower, y_preds_upper

    def plot_training_history(self):
        plt.figure(figsize=(12, 4))
        plt.suptitle("Upper and Lower Model Training Loss")
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
        plt.title('Upper Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def evaluate_model(self, y_preds_lower, y_preds_upper):
        
        def picp(y_pred_lower, y_pred_upper):
            """Prediction Interval Coverage Probability"""
            covered = np.sum((self.y_test >= y_pred_lower) & (self.y_test <= y_pred_upper))
            return covered / len(self.y_test)

        def mpiw(y_pred_lower, y_pred_upper):
            """Mean Prediction Interval Width"""
            return np.mean(y_pred_upper - y_pred_lower)

        
        return {
            'PICP': picp(y_preds_lower, y_preds_upper),
            'MPIW': mpiw(y_preds_lower, y_preds_upper)
        }

    def run_experiment(self, model, optimizer='adam', epochs=200, batch_size=32, verbose=0, model_param_string=None):
        y_preds_lower, y_preds_upper = self.train_model()
        self.plot_training_history()
        eval_dict = self.evaluate_model(y_preds_lower, y_preds_upper)

        print('\nModel Results - \n')
        print(f"PICP: {eval_dict['PICP'] * 100:.2f}%") 
        print(f"MPIW: {eval_dict['MPIW']:.4f}")

        