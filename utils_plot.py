#!/usr/bin/env python
"""
Collection of ____ Functions

This script contains a set of functions designed to ___

Functions included:
- lplot_feature_importance(estimator, feature_names, country): ____

Note:
Ensure that necessary libraries (______) are installed before running this script.

Author: Alexey Tyurin
Date: 1/20/2024
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.ticker as mticker

IMAGE_DIR = 'images'


def plot_feature_importance(estimator, feature_names, country):
    """
    plot feature importance
    """
    
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")

    feature_importance = estimator.best_estimator_["reg"].feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())

    data = pd.DataFrame({'name': feature_names, 'feature': feature_importance}).sort_values('feature')

    sns.barplot(data=data, x='feature', y='name', orient='h')
    ax.set_title(f'{country} - Variable Importance')
    ax.set_xlabel('Relative Importance')
    ax.set_ylabel('Variable')

    return

def create_learning_curves(X, y, models, country):
    """
    Create learning curves for multiple regression models.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), facecolor="white")
    axes = axes.ravel()  # Flatten the array for easy iteration

    for idx, (model_key, ax) in enumerate(zip(models.keys(), axes)):
        regressor = models[model_key]['regressor']
        best_params = models[model_key]['grid'].best_estimator_['reg'].get_params()

        if best_params is not None:
            regressor.set_params(**best_params)

        pipeline = Pipeline(steps=[("scaler", StandardScaler()), 
                                   ("reg", regressor)])
        
        train_sizes, train_scores, test_scores = learning_curve(pipeline, X, y,
                                                                cv=ShuffleSplit(n_splits=5, test_size=0.2, random_state=0), 
                                                                n_jobs=-1,
                                                                train_sizes=np.linspace(.1, 1.0, 10), 
                                                                scoring='r2') # 'neg_mean_squared_error', 'explained_variance'

        train_sizes_expanded = np.repeat(train_sizes, train_scores.shape[1])

        df_train = pd.DataFrame({'Train Size': train_sizes_expanded, 'Score': train_scores.flatten(), 'Set': 'Training'})
        df_test = pd.DataFrame({'Train Size': train_sizes_expanded, 'Score': test_scores.flatten(), 'Set': 'Cross-validation'})

        df = pd.concat([df_train, df_test])

        sns.lineplot(data=df, x='Train Size', y='Score', hue='Set', errorbar='sd', marker='o', palette=["red", "green"], ax=ax)

        ax.set_title(models[model_key]['name'])
        ax.set_xlabel('Training examples')
        ax.set_ylabel('R² (Coefficient of Determination)')
        ax.grid(True)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, pos: '{:.0%}'.format(val)))
        ax.legend(loc='best')        
        
    plt.suptitle(country)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, f'{country}_learning_curves.png'), format='png', dpi=300)
    plt.close()
