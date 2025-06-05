"""
AI-Powered Financial Fraud Detection System
Model Configuration

This module contains configuration parameters for the fraud detection models.

Author: Gabriel Demetrios Lafis
Date: June 2025
"""

# Random Forest parameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 15,
    'min_samples_leaf': 5,
    'class_weight': 'balanced',
    'n_jobs': -1,
    'random_state': 42
}

# XGBoost parameters
XGBOOST_PARAMS = {
    'learning_rate': 0.01,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': 10,
    'n_estimators': 300,
    'n_jobs': -1,
    'random_state': 42
}

# Neural Network parameters
NEURAL_NETWORK_PARAMS = {
    'hidden_layers': [128, 64, 32, 16],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 100,
    'random_state': 42
}

# Autoencoder parameters
AUTOENCODER_PARAMS = {
    'encoding_dim': 32,
    'hidden_layers': [64, 32],
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 100,
    'random_state': 42
}

# Meta-model parameters
META_MODEL_PARAMS = {
    'C': 1.0,
    'class_weight': 'balanced',
    'max_iter': 1000,
    'random_state': 42
}

# Feature engineering parameters
FEATURE_ENGINEERING_PARAMS = {
    'time_window_sizes': [1, 3, 7, 14, 30],  # Days
    'aggregation_functions': ['mean', 'std', 'min', 'max', 'sum', 'count'],
    'categorical_encoding': 'one-hot',
    'numerical_scaling': 'standard',
    'feature_selection_threshold': 0.01
}

# Model evaluation parameters
EVALUATION_PARAMS = {
    'cv_folds': 5,
    'test_size': 0.2,
    'metrics': ['auc_roc', 'precision', 'recall', 'f1_score', 'average_precision'],
    'threshold_optimization_metric': 'f1_score'
}

# Default model paths
MODEL_PATHS = {
    'random_forest': 'models/random_forest.pkl',
    'xgboost': 'models/xgboost.pkl',
    'neural_network': 'models/neural_network.h5',
    'autoencoder': 'models/autoencoder.h5',
    'meta_model': 'models/meta_model.pkl',
    'feature_scaler': 'models/feature_scaler.pkl',
    'ae_scaler': 'models/ae_scaler.pkl',
    'ensemble': 'models/ensemble'
}

