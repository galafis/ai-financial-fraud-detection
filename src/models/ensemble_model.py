
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
from utils.logger import get_logger
from config.model_config import (
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
        
        # Create models directory if it doesn\'t exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        logger.info("Initialized FraudDetectionEnsemble")
    
    def _build_random_forest(self) -> RandomForestClassifier:
        """Build and return a Random Forest classifier."""
        logger.info("Building Random Forest model")
        return RandomForestClassifier(
            n_estimators=self.rf_params.get("n_estimators", 500),
            max_depth=self.rf_params.get("max_depth", 15),
            min_samples_leaf=self.rf_params.get("min_samples_leaf", 5),
            class_weight=self.rf_params.get("class_weight", "balanced"),
            n_jobs=self.rf_params.get("n_jobs", -1),
            random_state=self.random_state
        )
    
    def _build_xgboost(self) -> xgb.XGBClassifier:
        """Build and return an XGBoost classifier."""
        logger.info("Building XGBoost model")
        return xgb.XGBClassifier(
            learning_rate=self.xgb_params.get("learning_rate", 0.01),
            max_depth=self.xgb_params.get("max_depth", 8),
            subsample=self.xgb_params.get("subsample", 0.8),
            colsample_bytree=self.xgb_params.get("colsample_bytree", 0.8),
            min_child_weight=self.xgb_params.get("min_child_weight", 3),
            gamma=self.xgb_params.get("gamma", 0.1),
            reg_alpha=self.xgb_params.get("reg_alpha", 0.1),
            reg_lambda=self.xgb_params.get("reg_lambda", 1.0),
            scale_pos_weight=self.xgb_params.get("scale_pos_weight", 10),
            n_estimators=self.xgb_params.get("n_estimators", 300),
            n_jobs=self.xgb_params.get("n_jobs", -1),
            random_state=self.random_state
        )
    
    def _build_neural_network(self) -> tf.keras.Model:
        """Build and return a Neural Network classifier."""
        logger.info("Building Neural Network model")
        
        # Get parameters
        input_dim = self.nn_params.get("input_dim", 100)
        hidden_layers = self.nn_params.get("hidden_layers", [128, 64, 32, 16])
        dropout_rate = self.nn_params.get("dropout_rate", 0.3)
        learning_rate = self.nn_params.get("learning_rate", 0.001)
        
        # Set random seed for TensorFlow
        tf.random.set_seed(self.random_state)
        
        # Build model
        model = Sequential()
        
        # Input layer
        model.add(Dense(hidden_layers[0], input_dim=input_dim, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation="relu"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation="sigmoid"))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def _build_autoencoder(self) -> tf.keras.Model:
        """Build and return an Autoencoder for anomaly detection."""
        logger.info("Building Autoencoder model")
        
        # Get parameters
        input_dim = self.ae_params.get("input_dim", 100)
        encoding_dim = self.ae_params.get("encoding_dim", 32)
        hidden_layers = self.ae_params.get("hidden_layers", [64, 32])
        learning_rate = self.ae_params.get("learning_rate", 0.001)
        
        # Set random seed for TensorFlow
        tf.random.set_seed(self.random_state)
        
        # Input layer
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoded = Dense(hidden_layers[0], activation="relu")(input_layer)
        encoded = BatchNormalization()(encoded)
        
        for units in hidden_layers[1:]:
            encoded = Dense(units, activation="relu")(encoded)
            encoded = BatchNormalization()(encoded)
        
        # Bottleneck
        bottleneck = Dense(encoding_dim, activation="relu")(encoded)
        
        # Decoder (symmetric to encoder)
        decoded = Dense(hidden_layers[-1], activation="relu")(bottleneck)
        decoded = BatchNormalization()(decoded)
        
        for units in reversed(hidden_layers[:-1]):
            decoded = Dense(units, activation="relu")(decoded)
            decoded = BatchNormalization()(decoded)
        
        # Output layer
        output_layer = Dense(input_dim, activation="linear")(decoded)
        
        # Create and compile model
        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="mean_squared_error"
        )
        
        return autoencoder
    
    def _build_meta_model(self) -> LogisticRegression:
        """Build and return a meta-model (Logistic Regression)."""
        logger.info("Building meta-model")
        return LogisticRegression(
            C=self.meta_params.get("C", 1.0),
            class_weight=self.meta_params.get("class_weight", "balanced"),
            max_iter=self.meta_params.get("max_iter", 1000),
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
            self.nn_params["input_dim"] = X_train.shape[1]
            self.neural_network = self._build_neural_network()
        
        if self.autoencoder is None:
            self.ae_params["input_dim"] = X_train.shape[1]
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
            verbose=False
        )
        
        # Fit Neural Network
        logger.info("Fitting Neural Network model")
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(self.models_dir, "nn_best_model.h5"),
                monitor="val_loss",
                save_best_only=True
            )
        ]
        
        self.neural_network.fit(
            X_train_scaled,
            y_train,
            validation_data=(X_val_scaled, y_val) if X_val is not None else None,
            epochs=self.nn_params.get("epochs", 100),
            batch_size=self.nn_params.get("batch_size", 256),
            callbacks=callbacks,
            verbose=0
        )
        
        # Load best Neural Network model if validation was used
        if X_val is not None:
            best_model_path = os.path.join(self.models_dir, "nn_best_model.h5")
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
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(self.models_dir, "autoencoder_best_model.h5"),
                monitor="val_loss",
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
            epochs=self.ae_params.get("epochs", 100),
            batch_size=self.ae_params.get("batch_size", 256),
            callbacks=ae_callbacks,
            verbose=0
        )
        
        # Load best Autoencoder model
        best_ae_model_path = os.path.join(self.models_dir, "autoencoder_best_model.h5")
        if os.path.exists(best_ae_model_path):
            self.autoencoder = load_model(best_ae_model_path)
        
        # Calibrate probabilities if specified
        if self.use_calibration:
            logger.info("Calibrating model probabilities")
            self.rf_calibrator = CalibratedClassifierCV(
                estimator=self.random_forest,
                cv=calibration_fold,
                method="isotonic"
            )
            self.rf_calibrator.fit(X_train_scaled, y_train)
            
            self.xgb_calibrator = CalibratedClassifierCV(
                estimator=self.xgboost,
                cv=calibration_fold,
                method="isotonic"
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
        """Get calibrated probabilities from Random Forest model.
        
        Args:
            X_scaled: Scaled features
            
        Returns:
            Array of probabilities
        """
        if self.use_calibration and self.rf_calibrator:
            return self.rf_calibrator.predict_proba(X_scaled)
        return self.random_forest.predict_proba(X_scaled)
    
    def _get_xgb_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        """Get calibrated probabilities from XGBoost model.
        
        Args:
            X_scaled: Scaled features
            
        Returns:
            Array of probabilities
        """
        if self.use_calibration and self.xgb_calibrator:
            return self.xgb_calibrator.predict_proba(X_scaled)
        return self.xgboost.predict_proba(X_scaled)
    
    def _get_nn_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        """Get probabilities from Neural Network model.
        
        Args:
            X_scaled: Scaled features
            
        Returns:
            Array of probabilities
        """
        return self.neural_network.predict(X_scaled).flatten()
    
    def _get_autoencoder_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Get anomaly scores from Autoencoder model.
        
        Args:
            X: Original features (not scaled)
            
        Returns:
            Array of anomaly scores
        """
        X_ae_scaled = self.ae_scaler.transform(X)
        reconstructions = self.autoencoder.predict(X_ae_scaled)
        reconstruction_errors = np.mean(np.square(X_ae_scaled - reconstructions), axis=1)
        return reconstruction_errors
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probabilities for new data.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of fraud probabilities
        """
        logger.info(f"Predicting probabilities for {len(X)} samples")
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        # Get predictions from base models
        rf_preds = self._get_rf_proba(X_scaled)[:, 1].reshape(-1, 1)
        xgb_preds = self._get_xgb_proba(X_scaled)[:, 1].reshape(-1, 1)
        nn_preds = self._get_nn_proba(X_scaled).reshape(-1, 1)
        ae_scores = self._get_autoencoder_scores(X).reshape(-1, 1)
        
        # Combine predictions for meta-model
        meta_features = np.hstack([rf_preds, xgb_preds, nn_preds, ae_scores])
        
        # Predict with meta-model
        if hasattr(self.meta_model, 'predict_proba'):
            # For classifiers that output probabilities (e.g., LogisticRegression)
            ensemble_proba = self.meta_model.predict_proba(meta_features)[:, 1]
        else:
            # For models that directly output scores or single-dimension predictions
            ensemble_proba = self.meta_model.predict(meta_features)
            
        return ensemble_proba
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud labels for new data.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of fraud labels (0 or 1)
        """
        logger.info(f"Predicting labels for {len(X)} samples")
        probabilities = self.predict_proba(X)
        return (probabilities >= self.threshold).astype(int)
    
    def _calculate_metrics(self, X: pd.DataFrame, y_true: pd.Series) -> Dict:
        """
        Calculate evaluation metrics for the model.
        
        Args:
            X: Features
            y_true: True labels
            
        Returns:
            Dictionary of metrics (ROC AUC, Precision, Recall, F1-Score)
        """
        y_proba = self.predict_proba(X)
        y_pred = (y_proba >= self.threshold).astype(int)
        
        roc_auc = roc_auc_score(y_true, y_proba)
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        f1 = f1_score(y_true, y_pred)
        
        # Find optimal threshold (e.g., maximizing F1-score)
        f1_scores = []
        thresholds = np.arange(0, 1, 0.01)
        for t in thresholds:
            y_pred_t = (y_proba >= t).astype(int)
            f1_scores.append(f1_score(y_true, y_pred_t))
        
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        
        return {
            "roc_auc": roc_auc,
            "f1_score": f1,
            "optimal_threshold": optimal_threshold
        }
        
    def explain_prediction(self, X_sample: pd.Series, method: str = "shap") -> Union[np.ndarray, Dict]:
        """
        Explain a single prediction using SHAP or LIME.
        
        Args:
            X_sample: A single sample (pd.Series) to explain
            method: Explanation method, either "shap" or "lime"
            
        Returns:
            SHAP values (np.ndarray) or LIME explanation (Dict)
        """
        if method == "shap":
            if self.shap_explainer is None:
                raise ValueError("SHAP explainer not initialized. Call fit() first.")
            
            # SHAP values for tree models (XGBoost)
            shap_values = self.shap_explainer.shap_values(self.feature_scaler.transform(X_sample.to_frame().T))
            return shap_values
        
        elif method == "lime":
            if self.lime_explainer is None:
                raise ValueError("LIME explainer not initialized. Call fit() first.")
            
            # LIME explanation
            explanation = self.lime_explainer.explain_instance(
                data_row=self.feature_scaler.transform(X_sample.to_frame().T)[0],
                predict_fn=self.predict_proba,
                num_features=len(X_sample)
            )
            return explanation.as_list()
        
        else:
            raise ValueError(f"Unknown explanation method: {method}. Choose \'shap\' or \'lime\'.")

    def save(self, path: str):
        """
        Save the ensemble model and its components.
        
        Args:
            path: Directory path to save the model artifacts.
        """
        logger.info(f"Saving model to {path}")
        os.makedirs(path, exist_ok=True)
        
        # Save configuration
        config = {
            "rf_params": self.rf_params,
            "xgb_params": self.xgb_params,
            "nn_params": self.nn_params,
            "ae_params": self.ae_params,
            "meta_params": self.meta_params,
            "use_calibration": self.use_calibration,
            "threshold": self.threshold,
            "random_state": self.random_state
        }
        with open(os.path.join(path, "config.pkl"), "wb") as f:
            pickle.dump(config, f)
            
        # Save individual models
        with open(os.path.join(path, "random_forest.pkl"), "wb") as f:
            pickle.dump(self.random_forest, f)
            
        with open(os.path.join(path, "xgboost.pkl"), "wb") as f:
            pickle.dump(self.xgboost, f)
            
        self.neural_network.save(os.path.join(path, "neural_network.h5"))
        self.autoencoder.save(os.path.join(path, "autoencoder.h5"))
        
        with open(os.path.join(path, "meta_model.pkl"), "wb") as f:
            pickle.dump(self.meta_model, f)
            
        # Save scalers
        with open(os.path.join(path, "feature_scaler.pkl"), "wb") as f:
            pickle.dump(self.feature_scaler, f)
            
        with open(os.path.join(path, "ae_scaler.pkl"), "wb") as f:
            pickle.dump(self.ae_scaler, f)
            
        # Save calibrators if they exist
        if self.rf_calibrator:
            with open(os.path.join(path, "rf_calibrator.pkl"), "wb") as f:
                pickle.dump(self.rf_calibrator, f)
        
        if self.xgb_calibrator:
            with open(os.path.join(path, "xgb_calibrator.pkl"), "wb") as f:
                pickle.dump(self.xgb_calibrator, f)
        
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str):
        """
        Load the ensemble model and its components.

        Args:
            path: Directory path where the model artifacts are saved.

        Returns:
            FraudDetectionEnsemble: Loaded ensemble model instance.

        Raises:
            FileNotFoundError: If model artifacts are not found.
        """
        logger.info(f"Loading model from {path}")

        # Load configuration
        with open(os.path.join(path, "config.pkl"), "rb") as f:
            config = pickle.load(f)

        instance = cls(
            random_forest_params=config["rf_params"],
            xgboost_params=config["xgb_params"],
            neural_network_params=config["nn_params"],
            autoencoder_params=config["ae_params"],
            meta_model_params=config["meta_params"],
            models_dir=path,
            use_calibration=config["use_calibration"],
            threshold=config["threshold"],
            random_state=config["random_state"]
        )

        # Load individual models
        with open(os.path.join(path, "random_forest.pkl"), "rb") as f:
            instance.random_forest = pickle.load(f)

        with open(os.path.join(path, "xgboost.pkl"), "rb") as f:
            instance.xgboost = pickle.load(f)

        instance.neural_network = load_model(os.path.join(path, "neural_network.h5"))
        instance.autoencoder = load_model(os.path.join(path, "autoencoder.h5"))

        with open(os.path.join(path, "meta_model.pkl"), "rb") as f:
            instance.meta_model = pickle.load(f)

        # Load calibrators if they exist
        rf_calibrator_path = os.path.join(path, "rf_calibrator.pkl")
        if os.path.exists(rf_calibrator_path):
            with open(rf_calibrator_path, "rb") as f:
                instance.rf_calibrator = pickle.load(f)
        
        xgb_calibrator_path = os.path.join(path, "xgb_calibrator.pkl")
        if os.path.exists(xgb_calibrator_path):
            with open(xgb_calibrator_path, "rb") as f:
                instance.xgb_calibrator = pickle.load(f)

        # Load scalers
        with open(os.path.join(path, "feature_scaler.pkl"), "rb") as f:
            instance.feature_scaler = pickle.load(f)
            
        with open(os.path.join(path, "ae_scaler.pkl"), "rb") as f:
            instance.ae_scaler = pickle.load(f)

        logger.info(f"Model loaded from {path}")
        return instance


