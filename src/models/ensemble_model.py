"""
AI-Powered Financial Fraud Detection System
Ensemble Model for Real-Time Fraud Detection

This module implements an advanced ensemble model combining multiple ML algorithms
for high-precision fraud detection in financial transactions.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import optuna
import shap
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionEnsemble:
    """
    Advanced ensemble model for financial fraud detection.
    
    Combines Random Forest, XGBoost, Neural Network, and Isolation Forest
    for high-precision fraud detection with explainability.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the ensemble model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or self._get_default_config()
        self.models = {}
        self.feature_importance = {}
        self.explainer = None
        self.is_trained = False
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for the ensemble."""
        return {
            'random_forest': {
                'n_estimators': 500,
                'max_depth': 20,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 'sqrt',
                'bootstrap': True,
                'oob_score': True,
                'n_jobs': -1,
                'random_state': 42,
                'class_weight': 'balanced'
            },
            'xgboost': {
                'n_estimators': 300,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'scale_pos_weight': 10,
                'random_state': 42
            },
            'neural_network': {
                'hidden_layers': [512, 256, 128, 64],
                'dropout_rates': [0.3, 0.4, 0.3, 0.2],
                'learning_rate': 0.001,
                'batch_size': 256,
                'epochs': 100,
                'patience': 10
            },
            'isolation_forest': {
                'n_estimators': 200,
                'contamination': 0.1,
                'random_state': 42,
                'n_jobs': -1
            },
            'ensemble_weights': {
                'random_forest': 0.3,
                'xgboost': 0.35,
                'neural_network': 0.25,
                'isolation_forest': 0.1
            }
        }
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """
        Train the ensemble model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training metrics and results
        """
        logger.info("Starting ensemble model training...")
        
        results = {}
        
        # Train Random Forest
        logger.info("Training Random Forest...")
        rf_results = self._train_random_forest(X_train, y_train)
        results['random_forest'] = rf_results
        
        # Train XGBoost with optimization
        logger.info("Training XGBoost with hyperparameter optimization...")
        xgb_results = self._train_xgboost_optimized(X_train, y_train)
        results['xgboost'] = xgb_results
        
        # Train Neural Network
        logger.info("Training Neural Network...")
        nn_results = self._train_neural_network(X_train, y_train, X_val, y_val)
        results['neural_network'] = nn_results
        
        # Train Isolation Forest
        logger.info("Training Isolation Forest...")
        if_results = self._train_isolation_forest(X_train)
        results['isolation_forest'] = if_results
        
        # Initialize SHAP explainer
        logger.info("Initializing SHAP explainer...")
        self._initialize_explainer(X_train.sample(min(1000, len(X_train))))
        
        self.is_trained = True
        logger.info("Ensemble training completed successfully!")
        
        return results
    
    def _train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train Random Forest model."""
        rf = RandomForestClassifier(**self.config['random_forest'])
        
        # Cross-validation
        cv_scores = cross_val_score(
            rf, X_train, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1'
        )
        
        # Train final model
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        
        # Feature importance
        self.feature_importance['random_forest'] = dict(
            zip(X_train.columns, rf.feature_importances_)
        )
        
        return {
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'oob_score': rf.oob_score_,
            'feature_importance': self.feature_importance['random_forest']
        }
    
    def _train_xgboost_optimized(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train XGBoost with Optuna optimization."""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 20),
                'random_state': 42
            }
            
            model = XGBClassifier(**params)
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                scoring='f1'
            )
            return cv_scores.mean()
        
        # Optimize hyperparameters
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        # Train final model with best parameters
        best_params = study.best_params
        xgb = XGBClassifier(**best_params)
        xgb.fit(X_train, y_train)
        self.models['xgboost'] = xgb
        
        # Feature importance
        self.feature_importance['xgboost'] = dict(
            zip(X_train.columns, xgb.feature_importances_)
        )
        
        return {
            'best_params': best_params,
            'best_cv_score': study.best_value,
            'feature_importance': self.feature_importance['xgboost']
        }
    
    def _train_neural_network(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """Train Neural Network model."""
        config = self.config['neural_network']
        
        # Build model
        model = Sequential()
        
        # Input layer
        model.add(Dense(config['hidden_layers'][0], input_dim=X_train.shape[1]))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(Dropout(config['dropout_rates'][0]))
        
        # Hidden layers
        for i, (units, dropout) in enumerate(zip(config['hidden_layers'][1:], 
                                                config['dropout_rates'][1:]), 1):
            model.add(Dense(units))
            model.add(BatchNormalization())
            model.add(tf.keras.layers.Activation('relu'))
            model.add(Dropout(dropout))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        optimizer = Adam(learning_rate=config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=config['patience'],
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Calculate class weights
        class_weights = {
            0: len(y_train) / (2 * (y_train == 0).sum()),
            1: len(y_train) / (2 * (y_train == 1).sum())
        }
        
        # Train model
        validation_data = (X_val, y_val) if X_val is not None else None
        history = model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=0
        )
        
        self.models['neural_network'] = model
        
        return {
            'final_loss': history.history['loss'][-1],
            'final_accuracy': history.history['accuracy'][-1],
            'training_epochs': len(history.history['loss'])
        }
    
    def _train_isolation_forest(self, X_train: pd.DataFrame) -> Dict:
        """Train Isolation Forest for anomaly detection."""
        iso_forest = IsolationForest(**self.config['isolation_forest'])
        iso_forest.fit(X_train)
        self.models['isolation_forest'] = iso_forest
        
        # Calculate anomaly scores
        anomaly_scores = iso_forest.decision_function(X_train)
        
        return {
            'mean_anomaly_score': anomaly_scores.mean(),
            'std_anomaly_score': anomaly_scores.std()
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of fraud probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = {}
        weights = self.config['ensemble_weights']
        
        # Random Forest predictions
        rf_proba = self.models['random_forest'].predict_proba(X)[:, 1]
        predictions['random_forest'] = rf_proba
        
        # XGBoost predictions
        xgb_proba = self.models['xgboost'].predict_proba(X)[:, 1]
        predictions['xgboost'] = xgb_proba
        
        # Neural Network predictions
        nn_proba = self.models['neural_network'].predict(X).flatten()
        predictions['neural_network'] = nn_proba
        
        # Isolation Forest predictions (convert to probabilities)
        iso_scores = self.models['isolation_forest'].decision_function(X)
        iso_proba = 1 / (1 + np.exp(iso_scores))  # Sigmoid transformation
        predictions['isolation_forest'] = iso_proba
        
        # Weighted ensemble
        ensemble_proba = (
            weights['random_forest'] * predictions['random_forest'] +
            weights['xgboost'] * predictions['xgboost'] +
            weights['neural_network'] * predictions['neural_network'] +
            weights['isolation_forest'] * predictions['isolation_forest']
        )
        
        return ensemble_proba
    
    def predict_with_explanation(self, X: pd.DataFrame, 
                                explain_top_n: int = 10) -> Tuple[np.ndarray, List[Dict]]:
        """
        Make predictions with SHAP explanations.
        
        Args:
            X: Features for prediction
            explain_top_n: Number of top features to explain
            
        Returns:
            Tuple of (predictions, explanations)
        """
        predictions = self.predict(X)
        
        if self.explainer is None:
            logger.warning("SHAP explainer not initialized. Returning predictions only.")
            return predictions, []
        
        # Get SHAP values for Random Forest (most interpretable)
        shap_values = self.explainer.shap_values(X)
        
        explanations = []
        for i, (pred, shap_vals) in enumerate(zip(predictions, shap_values[1])):
            # Get top contributing features
            feature_contributions = list(zip(X.columns, shap_vals))
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            explanation = {
                'prediction': float(pred),
                'top_features': [
                    {'feature': feat, 'contribution': float(contrib)}
                    for feat, contrib in feature_contributions[:explain_top_n]
                ]
            }
            explanations.append(explanation)
        
        return predictions, explanations
    
    def _initialize_explainer(self, X_sample: pd.DataFrame):
        """Initialize SHAP explainer with sample data."""
        try:
            self.explainer = shap.TreeExplainer(self.models['random_forest'])
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainer: {e}")
            self.explainer = None
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate the ensemble model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X_test)
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, predictions)
        classification_rep = classification_report(y_test, binary_predictions, output_dict=True)
        conf_matrix = confusion_matrix(y_test, binary_predictions)
        
        return {
            'auc_score': auc_score,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix.tolist(),
            'precision': classification_rep['1']['precision'],
            'recall': classification_rep['1']['recall'],
            'f1_score': classification_rep['1']['f1-score']
        }
    
    def save_model(self, filepath: str):
        """Save the trained ensemble model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'models': self.models,
            'config': self.config,
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load a trained ensemble model."""
        model_data = joblib.load(filepath)
        
        instance = cls(model_data['config'])
        instance.models = model_data['models']
        instance.feature_importance = model_data['feature_importance']
        instance.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
        return instance

# Example usage
if __name__ == "__main__":
    # This would typically be replaced with real data loading
    from sklearn.datasets import make_classification
    
    # Generate synthetic fraud detection dataset
    X, y = make_classification(
        n_samples=10000,
        n_features=50,
        n_informative=30,
        n_redundant=10,
        n_clusters_per_class=1,
        weights=[0.95, 0.05],  # Imbalanced dataset (5% fraud)
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, stratify=y_series, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    # Initialize and train ensemble
    ensemble = FraudDetectionEnsemble()
    training_results = ensemble.train(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    evaluation_results = ensemble.evaluate(X_test, y_test)
    
    print("Training Results:")
    for model_name, results in training_results.items():
        print(f"{model_name}: {results}")
    
    print("\nEvaluation Results:")
    print(f"AUC Score: {evaluation_results['auc_score']:.4f}")
    print(f"Precision: {evaluation_results['precision']:.4f}")
    print(f"Recall: {evaluation_results['recall']:.4f}")
    print(f"F1 Score: {evaluation_results['f1_score']:.4f}")
    
    # Save model
    ensemble.save_model('fraud_detection_ensemble.pkl')

