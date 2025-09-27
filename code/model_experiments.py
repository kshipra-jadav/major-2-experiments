from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

        
            
        


    def plot_prediction_interval_3d(self):
        # Import the necessary toolkit for 3D plotting
        # 1. Get the upper and lower predictions for the test set
        y_pred_upper_test, y_pred_lower_test, _ = self.evaluate_model(type='test')

        # 2. Define the axes for the surface
        # The X-axis is the index of each sample in the test set
        sample_indices = np.arange(len(self.y_test))
        # The Y-axis is a simple array to position the two bounds separately
        bound_axis = np.array([0, 1])

        # 3. Create a meshgrid from the two axes
        X_grid, Y_grid = np.meshgrid(sample_indices, bound_axis)

        # 4. Create the Z-grid, which contains the prediction values
        # The first row corresponds to the lower bound, the second to the upper bound
        Z_grid = np.vstack([y_pred_lower_test, y_pred_upper_test])

        # 5. Create the 3D plot
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 6. Plot the prediction interval as a single, continuous surface
        ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', alpha=0.8, edgecolor='none')

        # 7. Set the labels, title, and ticks
        ax.set_xlabel('\nSample Index', fontsize=12)
        ax.set_ylabel('\nPrediction Bound', fontsize=12)
        ax.set_zlabel('\nSoil Moisture', fontsize=12)
        ax.set_title('3D Prediction Interval Landscape', fontsize=16)

        # Use text labels for the Y-axis ticks instead of numbers
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Lower', 'Upper'], fontsize=10)
        
        # Adjust the viewing angle for a better perspective
        ax.view_init(elev=25, azim=-110)
        
        plt.show()