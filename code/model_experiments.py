from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)

class ModelExperiments:
    def __init__(self, X, y, test_size=0.2, val_size=0.2, random_state=10):
        self.random_state = random_state
        
        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for the remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_state
        )
        
        # Store the splits
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train.values.flatten()
        self.y_val = y_val.values.flatten()
        self.y_test = y_test.values.flatten()
        
        # Scale the data
        self.scaler_X = StandardScaler()
        
        self.X_train_scaled = self.scaler_X.fit_transform(X_train)
        self.X_val_scaled = self.scaler_X.transform(X_val)
        self.X_test_scaled = self.scaler_X.transform(X_test)
        
        
        print(f"Data split sizes:")
        print(f"Train: {len(X_train)} samples")
        print(f"Validation: {len(X_val)} samples")
        print(f"Test: {len(X_test)} samples")

    def fit_grid_search(self, model, param_grid, scaled=False, model_name="Model"):
        print(f"\n=== Running {model_name} ===")
        
        X_train = self.X_train_scaled if scaled else self.X_train
        X_val = self.X_val_scaled if scaled else self.X_val
        y_train = self.y_train
        y_val = self.y_val

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            verbose=1,
            scoring='neg_root_mean_squared_error'  # Better scoring for regression
        )

        grid_search.fit(X_train, y_train)

        print("Best Parameters:", grid_search.best_params_)

        # Evaluate on validation set
        best_model = grid_search.best_estimator_
        val_preds = best_model.predict(X_val)
        val_results = self.make_result_dict(y_val, val_preds, "Validation")
        
        # Also evaluate on test set for final comparison
        X_test = self.X_test_scaled if scaled else self.X_test
        y_test = self.y_test
        test_preds = best_model.predict(X_test)
        test_results = self.make_result_dict(y_test, test_preds, "Test")
        
        # Store the best model
        self.best_models = getattr(self, 'best_models', {})
        self.best_models[model_name] = best_model
        
        return {**val_results, **test_results}

    def make_result_dict(self, y_true, y_preds, set_name):
        """
        Create evaluation metrics dictionary with set name prefix
        """
        result_dict = {}
        prefix = f"{set_name}_"
        
        result_dict[prefix + 'MAE'] = round(mean_absolute_error(y_true, y_preds), 4)
        result_dict[prefix + 'MSE'] = round(mean_squared_error(y_true, y_preds), 4)
        result_dict[prefix + 'RMSE'] = round(root_mean_squared_error(y_true, y_preds), 4)
        result_dict[prefix + 'R2'] = round(r2_score(y_true, y_preds), 4)
        result_dict[prefix + 'MAPE'] = round(mean_absolute_percentage_error(y_true, y_preds), 4)
        
        return result_dict

    def run_all(self):
        """
        Runs all experiments: RF, XGB, AdaBoost, SVR
        """
        results = {}

        # Random Forest
        rf = RandomForestRegressor(random_state=self.random_state)
        rf_param_grid = {
            'n_estimators': [100, 200, 500],     
            'max_depth': [None, 5, 10, 20],      
            'min_samples_split': [2, 5, 10],     
            'min_samples_leaf': [1, 2, 4],       
            'max_features': ['sqrt', 'log2']     
        }
        results["RandomForest"] = self.fit_grid_search(rf, rf_param_grid, model_name="RandomForest")

        # XGBoost
        xgb = XGBRegressor(random_state=self.random_state, objective='reg:squarederror')
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
            estimator=DecisionTreeRegressor(random_state=self.random_state),
            random_state=self.random_state
        )
        ada_param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 1.0],
            'estimator__max_depth': [2, 3, 5, None],
            'estimator__min_samples_split': [2, 5, 10]
        }
        results["AdaBoost"] = self.fit_grid_search(ada, ada_param_grid, model_name="AdaBoost")

        # SVR (requires scaled data)
        svr = SVR()
        svr_param_grid = {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'epsilon': [0.01, 0.1, 0.2]
        }
        results["SVR"] = self.fit_grid_search(svr, svr_param_grid, scaled=True, model_name="SVR")

        return results

    def get_best_model(self, model_name):
        """Get the best trained model by name"""
        return self.best_models.get(model_name)