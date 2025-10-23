from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from mapie.regression import SplitConformalRegressor, CrossConformalRegressor
from mapie.utils import train_conformalize_test_split
from mapie.metrics.regression import regression_coverage_score, regression_mean_width_score, coverage_width_based

from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
from pathlib import Path

warnings.filterwarnings('ignore', category=FutureWarning)


# put this somewhere near your class (once)
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
        plot_dir = Path(f"/home/kshipra/work/major/ml experiments/output/plots/ML/{self.satellite}")
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
        plt.savefig(f'/home/kshipra/work/major/ml experiments/output/plots/ANN/{self.satellite}/{model_params}.png')
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
        super().__init__(data, features=features, target=target, test_size=test_size, 
                         val_size=val_size, random_state=random_state, satellite=satellite)
        
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

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
        early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
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
            'PICP': picp(y_true, y_pred_lower, y_pred_upper),
            'MPIW': mpiw(y_pred_lower, y_pred_upper)
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
        
        plot_dir = f"/home/kshipra/work/major/ml experiments/output/plots/{self.satellite}/PI"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.xlabel('Sample Index')
        plt.ylabel('Soil Moisture')
        plt.title(f'{self.satellite}: {model_param_string}\nPrediction Interval for Soil Moisture')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{plot_dir}/{model_param_string}.png", dpi=300)
        plt.show()

    def run_experiment(self, model, optimizer='adam', epochs=200, learning_rate=0.01, batch_size=32, verbose=0, model_param_string=None):
    # Unpack all four returned prediction arrays
        y_preds_lower_test, y_preds_upper_test, y_preds_lower_val, y_preds_upper_val = self.train_model(
            model, optimizer=optimizer, epochs=epochs, batch_size=batch_size, verbose=verbose, learning_rate=learning_rate
        )
        
        self.plot_training_history(model_param_string)
        
        # Pass all four arrays to the plotting function
        self.plot_prediction_interval(
            y_preds_lower_test, y_preds_upper_test,
            y_preds_lower_val, y_preds_upper_val,
            model_param_string
        )


class PredictionIntervalWithTubeLoss(ANNExperimentsV1):
    def __init__(self, data, features, target, test_size=0.1, val_size=0.25, random_state=42, satellite=None, q=0.95, r=0.5, delta=0):
        super().__init__(data, features=features, target=target, test_size=test_size, 
                         val_size=val_size, random_state=random_state, satellite=satellite)
        
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    

        # hyperparameters for Tube Loss
        self.q = q # target coverage # default: 0.95
        self.r= r  # for movement of PI tube # default: 0.5
        self.delta = delta #  for recalibration # default: 0


    def confidence_loss(self, y_true, y_pred):
        y_true = y_true[:, 0]
        f2 = y_pred[:, 0]
        f1 = y_pred[:, 1]

        c1 = (1 - self.q) * (f2 - y_true)
        c2 = (1 - self.q) * (y_true - f1)
        c3 = self.q * (f1 - y_true)
        c4 = self.q * (y_true - f2)

        # Use tf.where to create a tensor based on conditions
        loss_part1 = tf.where(y_true > self.r * (f1 + f2), c1, c2)
        loss_part2 = tf.where(f1 > y_true, c3, c4)

        final_loss = tf.where(tf.logical_and(y_true <= f2, y_true >= f1), loss_part1, loss_part2) + (self.delta * tf.abs(f1 - f2))

        # Reduce the loss to a scalar using tf.reduce_mean
        return tf.reduce_mean(final_loss)
    

    def train_model(self, model, optimizer, num_epochs=100, batch_size=32):
        # tf.keras.backend.clear_session() removing this since it messes with the seeds set.

        self.model = tf.keras.models.clone_model(model)
        self.model.compile(optimizer=optimizer, loss=self.confidence_loss)

        progress = EpochTqdm(total_epochs=num_epochs)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        self.history = self.model.fit(
            self.X_train_scaled, self.y_train,
            validation_data=(self.X_val_scaled, self.y_val),
            epochs=num_epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[progress, early_stopping]
        )
    
    def plot_losses(self):
        plt.figure(figsize=(12, 4))
        plt.title('Model Loss')
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self, type='test'):
        if type == 'test':
            y_preds = self.model.predict(self.X_test_scaled, verbose=0)
            y_test = np.stack((self.y_test, self.y_test), axis=1)
            
        if type == 'val':
            y_preds = self.model.predict(self.X_val_scaled, verbose=0)
            y_test = np.stack((self.y_val, self.y_val), axis=1)

        upper_preds = y_preds[:, 0]
        lower_preds = y_preds[:, 1]

        K_u = upper_preds > y_test[:, 0]
        K_l = lower_preds < y_test[:, 1]

        picp = np.mean(K_u * K_l)
        mpiw = np.round(np.mean(upper_preds - lower_preds), 3)

        results = {
            'PICP': picp,
            'MPIW': mpiw
        }
        return upper_preds, lower_preds, results

    def plot_prediction_interval(self, model_param_string, save_fig=False):
        indices = range(len(self.y_test))

        y_pred_upper_test, y_pred_lower_test, test_results = self.evaluate_model()
        _, _, val_results = self.evaluate_model(type='val')

        plt.figure(figsize=(16, 7))
        plt.plot(indices, self.y_test, 'o', color='blue', label='Actual Soil Moisture (Test Set)')
        plt.plot(indices, y_pred_lower_test, color='red', linestyle='--', label='Lower Bound')
        plt.plot(indices, y_pred_upper_test, color='orange', linestyle='--', label='Upper Bound')

        plt.fill_between(indices, y_pred_lower_test, y_pred_upper_test, color='gray', alpha=0.2, label='95% Prediction Interval')

        # Create a clean, aligned text box for both metrics
        test_metrics = f"Test  | PICP: {test_results['PICP']*100:5.2f}% | MPIW: {test_results['MPIW']:.4f}"
        val_metrics =  f"Valid | PICP: {val_results['PICP']*100:5.2f}% | MPIW: {val_results['MPIW']:.4f}"
        metrics_text = f"{test_metrics}\n{val_metrics}"
        
        plt.annotate(metrics_text, xy=(0.02, 0.98), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8), 
                    verticalalignment='top', fontsize=12,
                    # Using a monospaced font for clean alignment
                    fontname='monospace')
        savepath = f"/home/kshipra/work/major/ml experiments/output/plots/{self.satellite}/PI/TubeLoss/"
        os.makedirs(savepath, exist_ok=True)
        plt.xlabel('Sample Index')
        plt.ylabel('Soil Moisture')
        plt.title(f'Prediction Interval for Soil Moisture\n{self.satellite} - {model_param_string}\nr={self.r}   |   delta={self.delta}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_fig:
            plt.savefig(f"{savepath}/{model_param_string}.png", dpi=200, bbox_inches="tight")
        
        plt.show()
    
    def run_experiment(self, model, optimizer, epochs=200, batch_size=32, 
                       model_param_string=None, plot_losses=False, return_preds=False, 
                       save_fig=False, plot_interval=True):
        self.train_model(model=model, optimizer=optimizer, num_epochs=epochs, batch_size=batch_size)
        if plot_losses:
            self.plot_losses()
        if plot_interval:
            self.plot_prediction_interval(model_param_string=model_param_string, save_fig=save_fig)
        if return_preds:
            _, _, test_results = self.evaluate_model()
            _, _, val_results = self.evaluate_model('val')
            return {
                'test_results': test_results,
                'val_results': val_results
            }
    def plot_prediction_interval_3d(self, feature1_name, feature2_name, model_param_string):
        import plotly.graph_objects as go
        from scipy.interpolate import griddata

        X_original_test = self.X_test
        y_true_test = self.y_test
        y_pred_upper_test, y_pred_lower_test, test_results = self.evaluate_model()
        _, _, val_results = self.evaluate_model(type='val')

        # 2. Create a grid and interpolate (same as before)
        grid_res = 100
        x_grid = np.linspace(X_original_test[feature1_name].min(), X_original_test[feature1_name].max(), grid_res)
        y_grid = np.linspace(X_original_test[feature2_name].min(), X_original_test[feature2_name].max(), grid_res)
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)

        Z_upper_mesh = griddata(
            (X_original_test[feature1_name], X_original_test[feature2_name]),
            y_pred_upper_test, (X_mesh, Y_mesh), method='cubic'
        )
        Z_lower_mesh = griddata(
            (X_original_test[feature1_name], X_original_test[feature2_name]),
            y_pred_lower_test, (X_mesh, Y_mesh), method='cubic'
        )

        # 3. Create the interactive 3D plot with Plotly
        fig = go.Figure()

        # Add the Upper Bound Surface
        fig.add_trace(go.Surface(
            x=X_mesh, y=Y_mesh, z=Z_upper_mesh,
            opacity=0.5,
            colorscale=[[0, 'orange'], [1, 'orange']], # Solid color
            showscale=False,
            name='Upper Bound'
        ))

        # Add the Lower Bound Surface
        fig.add_trace(go.Surface(
            x=X_mesh, y=Y_mesh, z=Z_lower_mesh,
            opacity=0.5,
            colorscale=[[0, 'red'], [1, 'red']], # Solid color
            showscale=False,
            name='Lower Bound'
        ))

        # Add the Actual Data Points
        fig.add_trace(go.Scatter3d(
            x=X_original_test[feature1_name],
            y=X_original_test[feature2_name],
            z=y_true_test,
            mode='markers',
            marker=dict(size=3, color='blue'),
            name='Actual Soil Moisture'
        ))

        # 4. Create metrics text and layout
        test_metrics = f"Test  | PICP: {test_results['PICP']*100:5.2f}% | MPIW: {test_results['MPIW']:.4f}"
        val_metrics =  f"Valid | PICP: {val_results['PICP']*100:5.2f}% | MPIW: {val_results['MPIW']:.4f}"
        metrics_text = f"<b>Metrics</b><br>{test_metrics}<br>{val_metrics}"

        fig.update_layout(
            title=f'3D Prediction Interval for Soil Moisture<br>{self.satellite} - {model_param_string}',
            scene=dict(
                xaxis_title=f'Feature: {feature1_name}',
                yaxis_title=f'Feature: {feature2_name}',
                zaxis_title='Soil Moisture',
                # Aspect ratio can be adjusted for better viewing
                aspectratio=dict(x=1.5, y=1.5, z=1)
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(x=0.01, y=0.95),
            # Add the annotation text box
            annotations=[
                dict(
                    text=metrics_text,
                    x=0.01, y=0.98,
                    xref='paper', yref='paper',
                    align='left',
                    showarrow=False,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='black',
                    borderwidth=1,
                    font=dict(family='monospace', size=12)
                )
            ]
        )
        
        fig.show()
                
class ConformalRegression:
    def __init__(self, data, features, target, random_seed, train_size=0.6, conf_size=0.2, test_size=0.2):
        self.data = data
        self.features = features
        self.target = target
        self.train_size = train_size
        self.conf_size = conf_size
        self.test_size = test_size
        self.random_seed = random_seed

        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

        self._prepare_data()


    def _prepare_data(self):
        '''
        Split the data into train, calibration and test set
        '''
        (
            X_train, X_cal, X_test,
            y_train, y_cal, y_test
        ) = train_conformalize_test_split(self.data[self.features], self.data[self.target], 
                                          train_size=self.train_size, conformalize_size=self.conf_size, 
                                          test_size=self.test_size, random_state=self.random_seed)

        self.X_train = X_train
        self.X_conf = X_cal
        self.X_test = X_test

        self.y_train = y_train.values.ravel()
        self.y_conf = y_cal.values.ravel()
        self.y_test = y_test.values.ravel()

        total = len(self.data)
        print(
            "Split sizes:\n"
            f"  Train    - X: {getattr(self.X_train, 'shape', (len(self.X_train),))}, y: {len(self.y_train)} ({len(self.y_train)/total:.1%})\n"
            f"  Conformal- X: {getattr(self.X_conf,  'shape', (len(self.X_conf), ))}, y: {len(self.y_conf)} ({len(self.y_conf)/total:.1%})\n"
            f"  Test     - X: {getattr(self.X_test,  'shape', (len(self.X_test), ))}, y: {len(self.y_test)} ({len(self.y_test)/total:.1%})\n"
            f"  Original data rows: {total}"
        )
    
    def run_experiment(self, estimator_str: str = "RandomForest", confidence_level=0.95):
        estimator = None
        if estimator_str == "RandomForest":
            estimator = RandomForestRegressor(random_state=self.random_seed)
        elif estimator_str == "AdaBoost":
            estimator = AdaBoostRegressor(estimator=DecisionTreeRegressor(random_state=self.random_seed), 
                                          random_state=self.random_seed)
        elif estimator_str == "SVR":
            estimator = SVR()
        elif estimator_str == "XGBoost":
            estimator = XGBRegressor()
        
        mapie_regressor = SplitConformalRegressor(estimator=estimator, confidence_level=confidence_level,
                                                  prefit=False, verbose=1)
        
        mapie_regressor.fit(self.X_train, self.y_train)
        mapie_regressor.conformalize(self.X_conf, self.y_conf)

        y_preds, intervals = mapie_regressor.predict_interval(self.X_test)

        upper_interval = intervals[:, 0, 0]
        lower_interval = intervals[:, 1, 0]

        print(f"Y preds shape - {y_preds.shape}")
        print(f"interval shape - {intervals.shape}")
        print(f"Upper interval shape - {upper_interval.shape}")
        print(f"Lower interval shape - {lower_interval.shape}")

        self.lower_interval = lower_interval
        self.upper_interval = upper_interval
        self.intervals = intervals
        self.y_preds = y_preds

        self.evaluate_model(confidence_level)
        self.plot_prediction_intervals()

    def evaluate_model(self, confidence_level):
        """
        Calculates and prints key performance metrics for the conformal regression model.
        """
        print("\n--- Model Evaluation ---")
        
        coverage = regression_coverage_score(self.y_test, self.intervals)
        print(f"Effective Coverage: {coverage[0]:.3f} (target: {confidence_level})")
        
        width = regression_mean_width_score(self.intervals)
        print(f"Average Interval Width: {width[0]:.3f}")
        
        print("------------------------\n")

    def plot_prediction_intervals(self):
        """
        Generates a plot to visualize the prediction intervals against the true values.
        """
        # Sort values for a cleaner plot
        order = np.argsort(self.y_test)
        y_test_sorted = self.y_test[order]
        y_preds_sorted = self.y_preds[order]
        lower_interval_sorted = self.lower_interval[order]
        upper_interval_sorted = self.upper_interval[order]

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot the model's predictions
        ax.plot(y_test_sorted, y_preds_sorted, 'o', color='royalblue', label='Model Predictions', markersize=5)

        # Plot the prediction intervals as a shaded area
        ax.fill_between(
            y_test_sorted,
            lower_interval_sorted,
            upper_interval_sorted,
            alpha=0.2,
            color='royalblue',
            label='Prediction Interval'
        )
        
        ax.set_xlabel("True Values", fontsize=12)
        ax.set_ylabel("Predicted Values", fontsize=12)
        ax.set_title("Prediction Intervals vs. True Values", fontsize=14)
        ax.legend(loc="upper left")
        ax.grid(True)
        plt.show()


class NewConformalRegression:
    """
    An enhanced class to run, evaluate, and compare various conformal regression experiments.
    """
    def __init__(self, data, features, target, random_seed=42):
        self.data = data
        self.features = features
        self.target = target
        self.random_seed = random_seed
        self.results_ = {}

        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

    def _prepare_split_data(self, train_size=0.6, conf_size=0.2, test_size=0.2):
        """Prepares data for split-conformal methods."""
        (
            X_train, X_cal, X_test,
            y_train, y_cal, y_test
        ) = train_conformalize_test_split(
            self.data[self.features], self.data[self.target],
            train_size=train_size, conformalize_size=conf_size,
            test_size=test_size, random_state=self.random_seed
        )
        self.split_data = {
            "X_train": X_train, "y_train": y_train.values.ravel(),
            "X_conf": X_cal, "y_conf": y_cal.values.ravel(),
            "X_test": X_test, "y_test": y_test.values.ravel()
        }
        print("Data prepared for Split Conformal methods.")
        
    def _prepare_cv_data(self, train_size=0.8, test_size=0.2):
        """Prepares data for cross-conformal methods."""
        X = self.data[self.features]
        y = self.data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed
        )
        self.cv_data = {
            "X_train": X_train, "y_train": y_train.values.ravel(),
            "X_test": X_test, "y_test": y_test.values.ravel()
        }
        print("Data prepared for Cross-Conformal methods.")

    def run_experiments(self, experiments_to_run):
        """
        Run a series of conformal regression experiments.
        
        :param experiments_to_run: List of dictionaries, each specifying an experiment.
        """
        self._prepare_split_data()
        self._prepare_cv_data()
        
        for exp in experiments_to_run:
            print(f"\n--- Running Experiment: {exp['name']} ---")
            
            confidence = exp['confidence']
            estimator = exp['estimator']
            mapie_method = exp['mapie_method']
            
            # CORRECTED: Instantiate with confidence_level as per the correct API
            if mapie_method == SplitConformalRegressor:
                mapie = SplitConformalRegressor(estimator=estimator, confidence_level=confidence, prefit=False)
                data = self.split_data
                mapie.fit(data['X_train'], data['y_train'])
                mapie.conformalize(data['X_conf'], data['y_conf'])
                # CORRECTED: predict_interval does not take alpha/confidence_level
                y_preds, intervals = mapie.predict_interval(data['X_test'])

            elif mapie_method == CrossConformalRegressor:
                # CORRECTED: Instantiate with confidence_level and default cv
                mapie = CrossConformalRegressor(estimator=estimator, confidence_level=confidence)
                data = self.cv_data
                mapie.fit_conformalize(data['X_train'], data['y_train'])
                # CORRECTED: predict_interval does not take alpha/confidence_level
                y_preds, intervals = mapie.predict_interval(data['X_test'])

            # Evaluate and store results
            y_test = data['y_test']
            coverage = regression_coverage_score(y_test, intervals)
            width = regression_mean_width_score(intervals)
            
            self.results_[exp['name']] = {
                'y_preds': y_preds,
                'intervals': intervals,
                'y_test': y_test,
                'coverage': coverage[0],
                'width': width[0],
                'confidence': confidence
            }
            print(f"Coverage: {coverage[0]:.3f} (Target: {confidence}) | Width: {width[0]:.3f}")

    def plot_model_comparison(self):
        """Plots a bar chart comparing the coverage and width of all run experiments."""
        if not self.results_:
            print("No experiment results to plot. Please run experiments first.")
            return
            
        results_df = pd.DataFrame(self.results_).T.reset_index()
        results_df.rename(columns={'index': 'Model'}, inplace=True)

        fig, ax = plt.subplots(1, 2, figsize=(16, 7))
        sns.set_style("whitegrid")

        # Coverage Plot
        sns.barplot(x='coverage', y='Model', data=results_df, ax=ax[0], palette='viridis')
        for i, (cov, conf) in enumerate(zip(results_df['coverage'], results_df['confidence'])):
            ax[0].axvline(x=conf, color='r', linestyle='--', label=f'Target ({conf})' if i == 0 else "")
            ax[0].text(cov, i, f' {cov:.3f}', va='center', ha='left', color='white', weight='bold')
        ax[0].set_title('Effective Coverage vs. Target', fontsize=14)
        ax[0].set_xlabel('Coverage', fontsize=12)
        ax[0].set_xlim(left=min(results_df['coverage'].min(), results_df['confidence'].min()) * 0.95)
        ax[0].legend()

        # Width Plot
        sns.barplot(x='width', y='Model', data=results_df, ax=ax[1], palette='plasma')
        for i, width in enumerate(results_df['width']):
            ax[1].text(width, i, f' {width:.3f}', va='center', ha='left', color='white', weight='bold')
        ax[1].set_title('Average Interval Width', fontsize=14)
        ax[1].set_xlabel('Width', fontsize=12)
        
        plt.tight_layout()
        plt.show()

    def plot_interval_width_vs_prediction(self, experiment_name):
        """Plots interval width against the predicted value to check for adaptiveness."""
        if experiment_name not in self.results_:
            print(f"Error: Experiment '{experiment_name}' not found.")
            return

        res = self.results_[experiment_name]
        widths = res['intervals'][:, 1, 0] - res['intervals'][:, 0, 0]
        preds = res['y_preds']

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=preds, y=widths, alpha=0.6, label="Data points")
        sns.regplot(x=preds, y=widths, scatter=False, color='red', label="Trendline")
        plt.xlabel('Predicted Value', fontsize=12)
        plt.ylabel('Interval Width', fontsize=12)
        plt.title(f'Interval Width vs. Prediction for {experiment_name}', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_coverage_calibration(self, experiment_config):
        """
        Runs an experiment across multiple confidence levels and plots the calibration.
        """
        print(f"\n--- Running Calibration Study for {experiment_config['name']} ---")
        confidence_levels = np.arange(0.1, 1.0, 0.1)
        actual_coverages = []
        
        mapie_method_class = experiment_config['mapie_method']
        estimator = experiment_config['estimator']
        
        if mapie_method_class == SplitConformalRegressor:
            data = self.split_data
            base_mapie = mapie_method_class(estimator=estimator).fit(data['X_train'], data['y_train'])
            base_mapie.conformalize(data['X_conf'], data['y_conf'])
            y_test = data['y_test']
            
            for conf in confidence_levels:
                # We need to re-conformalize or re-predict with new confidence, 
                # but the new API requires re-init. A bit inefficient but correct.
                mapie_calibrated = SplitConformalRegressor(estimator=estimator, confidence_level=conf, prefit=True)
                mapie_calibrated.fit(data['X_train'], data['y_train'])
                mapie_calibrated.conformalize(data['X_conf'], data['y_conf'])
                _, intervals = mapie_calibrated.predict_interval(data['X_test'])
                actual_coverages.append(regression_coverage_score(y_test, intervals)[0])

        elif mapie_method_class == CrossConformalRegressor:
            data = self.cv_data
            y_test = data['y_test']

            for conf in confidence_levels:
                # Re-run for each confidence level
                mapie_calibrated = CrossConformalRegressor(estimator=estimator, confidence_level=conf)
                mapie_calibrated.fit_conformalize(data['X_train'], data['y_train'])
                _, intervals = mapie_calibrated.predict_interval(data['X_test'])
                actual_coverages.append(regression_coverage_score(y_test, intervals)[0])
        
        # Plotting
        plt.figure(figsize=(8, 8))
        plt.plot(confidence_levels, actual_coverages, "o-", label="Model Calibration")
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        plt.xlabel("Target Confidence Level", fontsize=12)
        plt.ylabel("Actual Coverage", fontsize=12)
        plt.title(f"Coverage Calibration Plot for {experiment_config['name']}", fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

