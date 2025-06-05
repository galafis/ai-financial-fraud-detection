"""
AI-Powered Financial Fraud Detection System
Ensemble Model Implementation

This module implements the ensemble model that combines multiple machine learning
models for fraud detection, including Random Forest, XGBoost, Neural Networks,
and Autoencoders for anomaly detection.

Author: Gabriel Demetrios Lafis
Date: June 2025
"""

import numpy as np
import pandas as pd
import pickle
import os
import logging
from typing import Dict, List, Tuple, Union, Optional

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Explainability
import shap
from lime import lime_tabular

# Local imports
from ..utils.logger import get_logger
from ..config.model_config import (
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
    NEURAL_NETWORK_PARAMS,
    AUTOENCODER_PARAMS,
    META_MODEL_PARAMS
)

logger = get_logger(__name__)


class FraudDetectionEnsemble:
    """
    Ensemble model for fraud detection that combines:
    1. Random Forest
    2. XGBoost
    3. Neural Network
    4. Autoencoder (for anomaly detection)
    
    The predictions from these models are combined using a meta-model (Logistic Regression).
    """
    
    def __init__(
        self,
        random_forest_params: Dict = None,
        xgboost_params: Dict = None,
        neural_network_params: Dict = None,
        autoencoder_params: Dict = None,
        meta_model_params: Dict = None,
        models_dir: str = "models",
        use_calibration: bool = True,
        threshold: float = 0.5,
        random_state: int = 42
    ):
        """
        Initialize the ensemble model with parameters for each base model.
        
        Args:
            random_forest_params: Parameters for Random Forest model
            xgboost_params: Parameters for XGBoost model
            neural_network_params: Parameters for Neural Network model
            autoencoder_params: Parameters for Autoencoder model
            meta_model_params: Parameters for meta-model (Logistic Regression)
            models_dir: Directory to save/load models
            use_calibration: Whether to calibrate probabilities
            threshold: Decision threshold for binary classification
            random_state: Random seed for reproducibility
        """
        self.models_dir = models_dir
        self.use_calibration = use_calibration
        self.threshold = threshold
        self.random_state = random_state
        
        # Initialize parameters with defaults if not provided
        self.rf_params = random_forest_params or RANDOM_FOREST_PARAMS
        self.xgb_params = xgboost_params or XGBOOST_PARAMS
        self.nn_params = neural_network_params or NEURAL_NETWORK_PARAMS
        self.ae_params = autoencoder_params or AUTOENCODER_PARAMS
        self.meta_params = meta_model_params or META_MODEL_PARAMS
        
        # Initialize models
        self.random_forest = None
        self.xgboost = None
        self.neural_network = None
        self.autoencoder = None
        self.meta_model = None
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        self.ae_scaler = StandardScaler()
        
        # Initialize calibrators
        self.rf_calibrator = None
        self.xgb_calibrator = None
        self.nn_calibrator = None
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        logger.info("Initialized FraudDetectionEnsemble")
    
    def _build_random_forest(self) -> RandomForestClassifier:
        """Build and return a Random Forest classifier."""
        logger.info("Building Random Forest model")
        return RandomForestClassifier(
            n_estimators=self.rf_params.get('n_estimators', 500),
            max_depth=self.rf_params.get('max_depth', 15),
            min_samples_leaf=self.rf_params.get('min_samples_leaf', 5),
            class_weight=self.rf_params.get('class_weight', 'balanced'),
            n_jobs=self.rf_params.get('n_jobs', -1),
            random_state=self.random_state
        )
    
    def _build_xgboost(self) -> xgb.XGBClassifier:
        """Build and return an XGBoost classifier."""
        logger.info("Building XGBoost model")
        return xgb.XGBClassifier(
            learning_rate=self.xgb_params.get('learning_rate', 0.01),
            max_depth=self.xgb_params.get('max_depth', 8),
            subsample=self.xgb_params.get('subsample', 0.8),
            colsample_bytree=self.xgb_params.get('colsample_bytree', 0.8),
            min_child_weight=self.xgb_params.get('min_child_weight', 3),
            gamma=self.xgb_params.get('gamma', 0.1),
            reg_alpha=self.xgb_params.get('reg_alpha', 0.1),
            reg_lambda=self.xgb_params.get('reg_lambda', 1.0),
            scale_pos_weight=self.xgb_params.get('scale_pos_weight', 10),
            n_estimators=self.xgb_params.get('n_estimators', 300),
            n_jobs=self.xgb_params.get('n_jobs', -1),
            random_state=self.random_state
        )
    
    def _build_neural_network(self) -> tf.keras.Model:
        """Build and return a Neural Network classifier."""
        logger.info("Building Neural Network model")
        
        # Get parameters
        input_dim = self.nn_params.get('input_dim', 100)
        hidden_layers = self.nn_params.get('hidden_layers', [128, 64, 32, 16])
        dropout_rate = self.nn_params.get('dropout_rate', 0.3)
        learning_rate = self.nn_params.get('learning_rate', 0.001)
        
        # Set random seed for TensorFlow
        tf.random.set_seed(self.random_state)
        
        # Build model
        model = Sequential()
        
        # Input layer
        model.add(Dense(hidden_layers[0], input_dim=input_dim, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def _build_autoencoder(self) -> tf.keras.Model:
        """Build and return an Autoencoder for anomaly detection."""
        logger.info("Building Autoencoder model")
        
        # Get parameters
        input_dim = self.ae_params.get('input_dim', 100)
        encoding_dim = self.ae_params.get('encoding_dim', 32)
        hidden_layers = self.ae_params.get('hidden_layers', [64, 32])
        learning_rate = self.ae_params.get('learning_rate', 0.001)
        
        # Set random seed for TensorFlow
        tf.random.set_seed(self.random_state)
        
        # Input layer
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoded = Dense(hidden_layers[0], activation='relu')(input_layer)
        encoded = BatchNormalization()(encoded)
        
        for units in hidden_layers[1:]:
            encoded = Dense(units, activation='relu')(encoded)
            encoded = BatchNormalization()(encoded)
        
        # Bottleneck
        bottleneck = Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder (symmetric to encoder)
        decoded = Dense(hidden_layers[-1], activation='relu')(bottleneck)
        decoded = BatchNormalization()(decoded)
        
        for units in reversed(hidden_layers[:-1]):
            decoded = Dense(units, activation='relu')(decoded)
            decoded = BatchNormalization()(decoded)
        
        # Output layer
        output_layer = Dense(input_dim, activation='linear')(decoded)
        
        # Create and compile model
        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mean_squared_error'
        )
        
        return autoencoder
    
    def _build_meta_model(self) -> LogisticRegression:
        """Build and return a meta-model (Logistic Regression)."""
        logger.info("Building meta-model")
        return LogisticRegression(
            C=self.meta_params.get('C', 1.0),
            class_weight=self.meta_params.get('class_weight', 'balanced'),
            max_iter=self.meta_params.get('max_iter', 1000),
            random_state=self.random_state
        )
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        calibration_fold: int = 5,
        fit_autoencoder_on_negatives_only: bool = True
    ) -> Dict:
        """
        Fit the ensemble model on training data.
        
        Args:
            X_train: Training features
            y_train: Training labels (0 for legitimate, 1 for fraud)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            calibration_fold: Number of folds for probability calibration
            fit_autoencoder_on_negatives_only: Whether to fit autoencoder only on legitimate transactions
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Fitting ensemble model on {len(X_train)} samples")
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.feature_scaler.transform(X_val)
        
        # Initialize models if not already initialized
        if self.random_forest is None:
            self.random_forest = self._build_random_forest()
        
        if self.xgboost is None:
            self.xgboost = self._build_xgboost()
        
        if self.neural_network is None:
            self.nn_params['input_dim'] = X_train.shape[1]
            self.neural_network = self._build_neural_network()
        
        if self.autoencoder is None:
            self.ae_params['input_dim'] = X_train.shape[1]
            self.autoencoder = self._build_autoencoder()
        
        # Fit Random Forest
        logger.info("Fitting Random Forest model")
        self.random_forest.fit(X_train_scaled, y_train)
        
        # Fit XGBoost
        logger.info("Fitting XGBoost model")
        self.xgboost.fit(
            X_train_scaled, 
            y_train,
            eval_set=[(X_val_scaled, y_val)] if X_val is not None else None,
            early_stopping_rounds=50 if X_val is not None else None,
            verbose=False
        )
        
        # Fit Neural Network
        logger.info("Fitting Neural Network model")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(self.models_dir, 'nn_best_model.h5'),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        self.neural_network.fit(
            X_train_scaled,
            y_train,
            validation_data=(X_val_scaled, y_val) if X_val is not None else None,
            epochs=self.nn_params.get('epochs', 100),
            batch_size=self.nn_params.get('batch_size', 256),
            callbacks=callbacks,
            verbose=0
        )
        
        # Load best Neural Network model if validation was used
        if X_val is not None:
            best_model_path = os.path.join(self.models_dir, 'nn_best_model.h5')
            if os.path.exists(best_model_path):
                self.neural_network = load_model(best_model_path)
        
        # Fit Autoencoder (on legitimate transactions only if specified)
        logger.info("Fitting Autoencoder model")
        if fit_autoencoder_on_negatives_only:
            # Get only legitimate transactions (label 0)
            X_train_legitimate = X_train[y_train == 0]
            X_train_ae_scaled = self.ae_scaler.fit_transform(X_train_legitimate)
        else:
            X_train_ae_scaled = self.ae_scaler.fit_transform(X_train)
        
        ae_callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(self.models_dir, 'autoencoder_best_model.h5'),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Split data for autoencoder validation
        ae_val_size = int(0.2 * len(X_train_ae_scaled))
        X_train_ae = X_train_ae_scaled[:-ae_val_size]
        X_val_ae = X_train_ae_scaled[-ae_val_size:]
        
        self.autoencoder.fit(
            X_train_ae,
            X_train_ae,  # Autoencoder tries to reconstruct the input
            validation_data=(X_val_ae, X_val_ae),
            epochs=self.ae_params.get('epochs', 100),
            batch_size=self.ae_params.get('batch_size', 256),
            callbacks=ae_callbacks,
            verbose=0
        )
        
        # Load best Autoencoder model
        best_ae_model_path = os.path.join(self.models_dir, 'autoencoder_best_model.h5')
        if os.path.exists(best_ae_model_path):
            self.autoencoder = load_model(best_ae_model_path)
        
        # Calibrate probabilities if specified
        if self.use_calibration:
            logger.info("Calibrating model probabilities")
            self.rf_calibrator = CalibratedClassifierCV(
                base_estimator=self.random_forest,
                cv=calibration_fold,
                method='isotonic'
            )
            self.rf_calibrator.fit(X_train_scaled, y_train)
            
            self.xgb_calibrator = CalibratedClassifierCV(
                base_estimator=self.xgboost,
                cv=calibration_fold,
                method='isotonic'
            )
            self.xgb_calibrator.fit(X_train_scaled, y_train)
        
        # Generate predictions from base models for meta-model training
        rf_preds = self._get_rf_proba(X_train_scaled)[:, 1].reshape(-1, 1)
        xgb_preds = self._get_xgb_proba(X_train_scaled)[:, 1].reshape(-1, 1)
        nn_preds = self._get_nn_proba(X_train_scaled).reshape(-1, 1)
        
        # Get autoencoder reconstruction error as anomaly score
        ae_scores = self._get_autoencoder_scores(X_train).reshape(-1, 1)
        
        # Combine predictions for meta-model
        meta_features = np.hstack([rf_preds, xgb_preds, nn_preds, ae_scores])
        
        # Fit meta-model
        logger.info("Fitting meta-model")
        if self.meta_model is None:
            self.meta_model = self._build_meta_model()
        
        self.meta_model.fit(meta_features, y_train)
        
        # Initialize explainers
        logger.info("Initializing explainers")
        self.shap_explainer = shap.TreeExplainer(self.xgboost)
        
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train_scaled,
            feature_names=X_train.columns.tolist(),
            class_names=["Legitimate", "Fraudulent"],
            mode="classification",
            random_state=self.random_state
        )
        
        # Calculate and return training metrics
        metrics = self._calculate_metrics(X_train, y_train)
        logger.info(f"Training metrics: {metrics}")
        
        return metrics
    
    def _get_rf_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        """Get calibrated probabilities from Random Forest model."""
        if self.use_calibration and self.rf_calibrator is not None:
            return self.rf_calibrator.predict_proba(X_scaled)
        return self.random_forest.predict_proba(X_scaled)
    
    def _get_xgb_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        """Get calibrated probabilities from XGBoost model."""
        if self.use_calibration and self.xgb_calibrator is not None:
            return self.xgb_calibrator.predict_proba(X_scaled)
        return self.xgboost.predict_proba(X_scaled)
    
    def _get_nn_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        """Get probabilities from Neural Network model."""
        return self.neural_network.predict(X_scaled, verbose=0)
    
    def _get_autoencoder_scores(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get anomaly scores from Autoencoder based on reconstruction error.
        Higher score means more anomalous (potentially fraudulent).
        """
        X_scaled = self.ae_scaler.transform(X)
        reconstructed = self.autoencoder.predict(X_scaled, verbose=0)
        # Mean squared error as reconstruction error
        mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
        # Min-max scale to [0, 1] range
        min_val, max_val = np.min(mse), np.max(mse)
        if max_val > min_val:
            return (mse - min_val) / (max_val - min_val)
        return np.zeros_like(mse)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probability for input samples.
        
        Args:
            X: Input features
            
        Returns:
            Array of fraud probabilities
        """
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        # Get predictions from base models
        rf_preds = self._get_rf_proba(X_scaled)[:, 1].reshape(-1, 1)
        xgb_preds = self._get_xgb_proba(X_scaled)[:, 1].reshape(-1, 1)
        nn_preds = self._get_nn_proba(X_scaled).reshape(-1, 1)
        ae_scores = self._get_autoencoder_scores(X).reshape(-1, 1)
        
        # Combine predictions for meta-model
        meta_features = np.hstack([rf_preds, xgb_preds, nn_preds, ae_scores])
        
        # Get meta-model predictions
        return self.meta_model.predict_proba(meta_features)[:, 1]
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud (1) or legitimate (0) for input samples.
        
        Args:
            X: Input features
            
        Returns:
            Array of binary predictions (0 for legitimate, 1 for fraud)
        """
        probas = self.predict_proba(X)
        return (probas >= self.threshold).astype(int)
    
    def predict_with_details(self, X: pd.DataFrame) -> Dict:
        """
        Predict with detailed information including base model predictions and explanations.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with predictions, probabilities, and explanations
        """
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        # Get predictions from base models
        rf_preds = self._get_rf_proba(X_scaled)[:, 1]
        xgb_preds = self._get_xgb_proba(X_scaled)[:, 1]
        nn_preds = self._get_nn_proba(X_scaled).flatten()
        ae_scores = self._get_autoencoder_scores(X)
        
        # Combine predictions for meta-model
        meta_features = np.hstack([
            rf_preds.reshape(-1, 1),
            xgb_preds.reshape(-1, 1),
            nn_preds.reshape(-1, 1),
            ae_scores.reshape(-1, 1)
        ])
        
        # Get meta-model predictions
        meta_probas = self.meta_model.predict_proba(meta_features)[:, 1]
        meta_preds = (meta_probas >= self.threshold).astype(int)
        
        # Generate explanations for the first sample
        if len(X) > 0:
            # SHAP explanation
            shap_values = self.shap_explainer.shap_values(X_scaled[0:1])
            
            # Get top features and their importance
            feature_names = X.columns.tolist()
            feature_importance = np.abs(shap_values[0])
            top_indices = np.argsort(feature_importance)[-5:]  # Top 5 features
            top_features = [feature_names[i] for i in top_indices]
            top_importance = feature_importance[top_indices].tolist()
            
            # LIME explanation
            lime_exp = self.lime_explainer.explain_instance(
                X_scaled[0],
                lambda x: self.meta_model.predict_proba(
                    np.hstack([
                        self._get_rf_proba(x)[:, 1].reshape(-1, 1),
                        self._get_xgb_proba(x)[:, 1].reshape(-1, 1),
                        self._get_nn_proba(x).reshape(-1, 1),
                        np.zeros((len(x), 1))  # Placeholder for AE scores
                    ])
                ),
                num_features=5
            )
            
            lime_features = [feature[0] for feature in lime_exp.as_list()]
            lime_importance = [feature[1] for feature in lime_exp.as_list()]
            
            explanation = {
                "shap": {
                    "features": top_features,
                    "importance": top_importance
                },
                "lime": {
                    "features": lime_features,
                    "importance": lime_importance
                }
            }
        else:
            explanation = {}
        
        return {
            "predictions": meta_preds.tolist(),
            "probabilities": meta_probas.tolist(),
            "base_models": {
                "random_forest": rf_preds.tolist(),
                "xgboost": xgb_preds.tolist(),
                "neural_network": nn_preds.tolist(),
                "autoencoder": ae_scores.tolist()
            },
            "explanation": explanation
        }
    
    def _calculate_metrics(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Calculate performance metrics on the given data."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Calculate AUC-ROC
        auc_roc = roc_auc_score(y, y_proba)
        
        # Calculate F1 score
        f1 = f1_score(y, y_pred)
        
        # Calculate precision and recall at threshold
        precision, recall, thresholds = precision_recall_curve(y, y_proba)
        threshold_idx = np.argmin(np.abs(thresholds - self.threshold)) if len(thresholds) > 0 else 0
        precision_at_threshold = precision[threshold_idx] if threshold_idx < len(precision) else precision[-1]
        recall_at_threshold = recall[threshold_idx] if threshold_idx < len(recall) else recall[-1]
        
        return {
            "auc_roc": auc_roc,
            "f1_score": f1,
            "precision": precision_at_threshold,
            "recall": recall_at_threshold,
            "threshold": self.threshold
        }
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating model on {len(X)} samples")
        metrics = self._calculate_metrics(X, y)
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def save(self, path: str = None) -> None:
        """
        Save the ensemble model to disk.
        
        Args:
            path: Directory to save the model (defaults to self.models_dir)
        """
        if path is None:
            path = self.models_dir
        
        os.makedirs(path, exist_ok=True)
        
        # Save individual models
        with open(os.path.join(path, 'random_forest.pkl'), 'wb') as f:
            pickle.dump(self.random_forest, f)
        
        with open(os.path.join(path, 'xgboost.pkl'), 'wb') as f:
            pickle.dump(self.xgboost, f)
        
        self.neural_network.save(os.path.join(path, 'neural_network.h5'))
        self.autoencoder.save(os.path.join(path, 'autoencoder.h5'))
        
        with open(os.path.join(path, 'meta_model.pkl'), 'wb') as f:
            pickle.dump(self.meta_model, f)
        
        # Save calibrators if they exist
        if self.rf_calibrator is not None:
            with open(os.path.join(path, 'rf_calibrator.pkl'), 'wb') as f:
                pickle.dump(self.rf_calibrator, f)
        
        if self.xgb_calibrator is not None:
            with open(os.path.join(path, 'xgb_calibrator.pkl'), 'wb') as f:
                pickle.dump(self.xgb_calibrator, f)
        
        # Save scalers
        with open(os.path.join(path, 'feature_scaler.pkl'), 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        with open(os.path.join(path, 'ae_scaler.pkl'), 'wb') as f:
            pickle.dump(self.ae_scaler, f)
        
        # Save configuration
        config = {
            'rf_params': self.rf_params,
            'xgb_params': self.xgb_params,
            'nn_params': self.nn_params,
            'ae_params': self.ae_params,
            'meta_params': self.meta_params,
            'use_calibration': self.use_calibration,
            'threshold': self.threshold,
            'random_state': self.random_state
        }
        
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(config, f)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'FraudDetectionEnsemble':
        """
        Load the ensemble model from disk.
        
        Args:
            path: Directory to load the model from
            
        Returns:
            Loaded FraudDetectionEnsemble instance
        """
        # Load configuration
        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        
        # Create instance with loaded configuration
        instance = cls(
            random_forest_params=config['rf_params'],
            xgboost_params=config['xgb_params'],
            neural_network_params=config['nn_params'],
            autoencoder_params=config['ae_params'],
            meta_model_params=config['meta_params'],
            models_dir=path,
            use_calibration=config['use_calibration'],
            threshold=config['threshold'],
            random_state=config['random_state']
        )
        
        # Load individual models
        with open(os.path.join(path, 'random_forest.pkl'), 'rb') as f:
            instance.random_forest = pickle.load(f)
        
        with open(os.path.join(path, 'xgboost.pkl'), 'rb') as f:
            instance.xgboost = pickle.load(f)
        
        instance.neural_network = load_model(os.path.join(path, 'neural_network.h5'))
        instance.autoencoder = load_model(os.path.join(path, 'autoencoder.h5'))
        
        with open(os.path.join(path, 'meta_model.pkl'), 'rb') as f:
            instance.meta_model = pickle.load(f)
        
        # Load calibrators if they exist
        rf_calibrator_path = os.path.join(path, 'rf_calibrator.pkl')
        if os.path.exists(rf_calibrator_path):
            with open(rf_calibrator_path, 'rb') as f:
                instance.rf_calibrator = pickle.load(f)
        
        xgb_calibrator_path = os.path.join(path, 'xgb_calibrator.pkl')
        if os.path.exists(xgb_calibrator_path):
            with open(xgb_calibrator_path, 'rb') as f:
                instance.xgb_calibrator = pickle.load(f)
        
        # Load scalers
        with open(os.path.join(path, 'feature_scaler.pkl'), 'rb') as f:
            instance.feature_scaler = pickle.load(f)
        
        with open(os.path.join(path, 'ae_scaler.pkl'), 'rb') as f:
            instance.ae_scaler = pickle.load(f)
        
        # Initialize explainers
        instance.shap_explainer = shap.TreeExplainer(instance.xgboost)
        
        logger.info(f"Model loaded from {path}")
        return instance


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Load sample data (replace with actual data loading)
    # data = pd.read_csv("data/processed/transactions.csv")
    # X = data.drop("is_fraud", axis=1)
    # y = data["is_fraud"]
    
    # For demonstration, create synthetic data
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.99, 0.01],  # Imbalanced dataset
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train model
    model = FraudDetectionEnsemble(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    print(f"Test metrics: {metrics}")
    
    # Make predictions
    sample = X_test.iloc[:5]
    predictions = model.predict_with_details(sample)
    print(f"Predictions: {predictions}")
    
    # Save model
    model.save("models/ensemble_model")
    
    # Load model
    loaded_model = FraudDetectionEnsemble.load("models/ensemble_model")
    
    # Verify loaded model
    loaded_metrics = loaded_model.evaluate(X_test, y_test)
    print(f"Loaded model metrics: {loaded_metrics}")

