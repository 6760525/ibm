#!/usr/bin/env python
"""
Collection of Data Modeling Functions

This script contains a set of functions designed to ___________

Functions included:
- model_load(model_dir=MODEL_DIR, dev=DEV, verbose=True): ______

Note:
Ensure that necessary libraries (______) are installed before running this script.

Author: Alexey Tyurin
Date: 1/20/2024
"""

# Libraries

import os, re, joblib, time
import pandas as pd
import numpy as np
from utils_logger import *
from utils_data import *
from utils_plot import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore')

MODEL_DIR = os.path.join(".", "models")
DEV = True
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = 'init model'

INIT_MODELS = {
    'LNR': {'name': 'Linear Regressor',
            'regressor': LinearRegression(),
            'params': {'reg__fit_intercept': [True, False]},
            'grid': None},
    'KNR': {'name': 'K Neighbors Regressor',
            'regressor': KNeighborsRegressor(),
            'params': {'reg__n_neighbors': [3, 5, 7, 9, 11],
                       'reg__weights': ['uniform', 'distance'],
                       'reg__metric': ['euclidean', 'manhattan']},
            'grid': None},
    'RFR': {'name': 'Random Forest Regressor',
            'regressor': RandomForestRegressor(random_state=42),
            'params': {'reg__n_estimators': [10, 30, 50],
                       'reg__max_features': [3, 4, 5],
                       'reg__bootstrap': [True, False]},
            'grid': None},
    'SGD': {'name': 'Stochastic Gradient Regressor',
            'regressor': SGDRegressor(random_state=42),
            'params': {'reg__penalty': ['l1', 'l2', 'elasticnet'],
                       'reg__learning_rate': ['constant', 'optimal', 'invscaling']},
            'grid': None},
    'GBR': {'name': 'Gradient Boosting Regressor',
            'regressor': GradientBoostingRegressor(random_state=42),
            'params': {'reg__n_estimators': [10, 30, 50],
                       'reg__max_features': [3, 4, 5],
                       'reg__learning_rate': [1, 0.1, 0.01, 0.001]},
            'grid': None},
    'ABR': {'name': 'Ada Boosting Regressor',
            'regressor': AdaBoostRegressor(random_state=42),
            'params': {'reg__n_estimators': [10, 30, 50],
                       'reg__learning_rate': [1, 0.1, 0.01, 0.001]},
            'grid': None},
    }


def model_load(model_dir=MODEL_DIR, dev=DEV, verbose=True):
    """
    load models
    """
    
    if verbose:
        print("Loading Models")
    
    if dev:
        prefix = "test"
    else:
        prefix = "prod"
    
    if not os.path.exists(MODEL_DIR):
        raise Exception("Opps! Model dir does not exist")
    
    ## list model files from model directory
    models = [f for f in os.listdir(model_dir) if re.search(prefix, f)]

    if len(models) == 0:
        raise Exception(f"Models with prefix {prefix} cannot be found did you train?")
    
    ## load models
    all_models = {}
    for model in models:
        all_models[re.split("-", model)[1]] = joblib.load(os.path.join(model_dir, model))
        
    return(all_models)


def model_train(save_img=False,dev=DEV, verbose=True):
    """
    train models
    """
    
    ## load engineered features
    all_features = data_for_training(dev=dev, training=True, verbose=verbose)
    
    if verbose:
        print("Training Models")
    
    # Train models for each country
    for country in all_features.keys():
        time_start = time.time()
        models = INIT_MODELS.copy()

        X = all_features[country]['X']
        y = all_features[country]['y']
        feature_names = all_features[country]['features']

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

        min_rmse = np.inf

        for model in models:
            pipe = Pipeline(steps=[('scaler', StandardScaler()),
                                ('reg', models[model]['regressor'])])
                            
            grid = GridSearchCV(
                            pipe,
                            param_grid=models[model]['params'],
                            scoring='neg_mean_squared_error',
                            cv=5,
                            n_jobs=-1,
                            return_train_score=True)
                    
            grid.fit(X_train, y_train)

            models[model]['grid'] = grid

            y_pred = grid.predict(X_valid)
            rmse = np.sqrt(mean_squared_error(y_pred, y_valid))
            r2 = r2_score(y_pred, y_valid)
            if rmse < min_rmse:
                min_rmse = rmse
                min_r2 = r2
                opt_model = model

        create_learning_curves(X, y, models, country)
        print(f"Country: {country} ...best model: {models[opt_model]['name']}, rmse = {min_rmse:,.2f}, R^2 = {min_r2:.1%}")

        models[opt_model]['grid'].fit(X, y)

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        
        saved_model = os.path.join(MODEL_DIR, f"test-{country}-model-{str(MODEL_VERSION).replace('.', '_')}.joblib")
        joblib.dump(models[opt_model]['grid'], saved_model)

        if 'feature_importances_' in dir(models[opt_model]['grid'].best_estimator_["reg"]):
            plot_feature_importance(models[opt_model]['grid'], feature_names=feature_names, country=country)
        else:
            plot_feature_importance(models["RFR"]['grid'], feature_names=feature_names, country=country)
        
        plt.subplots_adjust(left=0.15)  # You can increase the value if more space is needed
        plt.tight_layout(pad=1.0)
        plt.savefig(os.path.join(IMAGE_DIR, f'{country}_features_importance.png'), format='png', dpi=300)
        plt.close()
        
        m, s = divmod(time.time()-time_start, 60)
        h, m = divmod(m, 60)
        runtime = "%03d:%02d:%02d"%(h, m, s)
                
        ## update log
        update_train_log(tag=country,
                        algorithm=models[model]['name'],
                        score={'rmse': min_rmse, 'r2': min_r2},
                        runtime=runtime,
                        model_version=MODEL_VERSION,
                        model_note=MODEL_VERSION_NOTE,
                        dev=DEV,
                        verbose=False)
    return


def model_predict(year, month, day, country, dev=DEV, verbose=True):
    """
    Make predictions for a specified date and country.
    """
    time_start = time.time()
    
    # Validate input date
    if not all(isinstance(d, int) for d in [year, month, day]):
        raise ValueError("Year, month, and day must be integers.")
        
    # Prepare the target date string
    target_date_str = f"{year}-{month:02d}-{day:02d}"
    if verbose:
        print(f"Make Prediction for {target_date_str}")

    # Load datasets and models
    datasets = data_for_training(training=False, dev=dev, verbose=verbose)
    models = model_load(dev=dev, verbose=verbose)

    # Check availability of models and datasets for the given country
    if country not in models or country not in datasets:
        raise Exception(f"Model or dataset for country '{country}' could not be found")
    
    # Extract the dataset and model for the given country
    X = datasets[country]['X']
    dates = datasets[country]['invoice_dates']
    labels = datasets[country]['features']
    model = models[country]
    df = pd.DataFrame(X, columns=labels, index=dates)

    # Check if the target date is within the dataset's range
    if target_date_str not in df.index.strftime('%Y-%m-%d'):
        first_date = df.index[0].strftime('%Y-%m-%d')
        last_date = df.index[-1].strftime('%Y-%m-%d')
        raise Exception(f"{target_date_str} not in range {first_date} and {last_date}")
    
    # Query and predict
    X_pred = df.loc[target_date_str].values.reshape(1, -1)
    y_pred = model.predict(X_pred)
    
    # Calculate runtime
    runtime = time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
    
    # Update predict log
    update_predict_log(country.upper(), y_pred, target_date_str, runtime, MODEL_VERSION, MODEL_VERSION_NOTE, dev=dev, verbose=verbose)
    
    return {"y_pred": y_pred}
