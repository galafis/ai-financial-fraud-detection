"""
AI-Powered Financial Fraud Detection System
Model Monitoring Module

This module implements monitoring for the fraud detection model,
including drift detection, performance metrics, and alerting.

Author: Gabriel Demetrios Lafis
Date: June 2025
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional, Any
import json
import os
import logging
import time
from collections import deque
import threading
import pickle
from scipy import stats

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Monitoring libraries
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, confusion_matrix
)

# Local imports
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelMonitor:
    """
    Monitor for fraud detection model performance and data drift.
    
    This class implements:
    1. Performance metrics tracking
    2. Data drift detection
    3. Concept drift detection
    4. Alerting for degraded performance
    5. Visualization of monitoring metrics
    """
    
    def __init__(
        self,
        metrics_file: str = "monitoring/metrics.json",
        reference_data: Optional[pd.DataFrame] = None,
        performance_window: int = 1000,
        drift_threshold: float = 0.05,
        alert_threshold: float = 0.1
    ):
        """
        Initialize the model monitor.
        
        Args:
            metrics_file: File to store monitoring metrics
            reference_data: Reference data for drift detection
            performance_window: Number of predictions to keep for performance metrics
            drift_threshold: Threshold for drift detection
            alert_threshold: Threshold for performance alerts
        """
        self.metrics_file = metrics_file
        self.reference_data = reference_data
        self.performance_window = performance_window
        self.drift_threshold = drift_threshold
        self.alert_threshold = alert_threshold
        
        # Create directory for metrics file
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        
        # Initialize metrics storage
        self.metrics_history = self._load_metrics()
        
        # Initialize performance tracking
        self.predictions = deque(maxlen=performance_window)
        self.labels = deque(maxlen=performance_window)
        self.timestamps = deque(maxlen=performance_window)
        
        # Initialize feature distributions
        self.reference_distributions = {}
        if reference_data is not None:
            self._compute_reference_distributions()
        
        # Initialize lock for thread safety
        self.lock = threading.Lock()
        
        logger.info("Initialized ModelMonitor")
    
    def _load_metrics(self) -> Dict:
        """
        Load metrics from file.
        
        Returns:
            Dictionary of metrics
        """
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metrics: {str(e)}")
        
        # Return empty metrics if file doesn't exist or error occurs
        return {
            "performance": {
                "auc_roc": [],
                "precision": [],
                "recall": [],
                "f1_score": [],
                "average_precision": [],
                "timestamps": []
            },
            "drift": {
                "feature_drift": {},
                "concept_drift": [],
                "timestamps": []
            },
            "alerts": []
        }
    
    def _save_metrics(self) -> None:
        """Save metrics to file."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
    
    def _compute_reference_distributions(self) -> None:
        """Compute reference distributions for drift detection."""
        if self.reference_data is None:
            logger.warning("No reference data provided for drift detection")
            return
        
        logger.info("Computing reference distributions")
        
        for column in self.reference_data.columns:
            try:
                # Skip non-numeric columns
                if not pd.api.types.is_numeric_dtype(self.reference_data[column]):
                    continue
                
                # Compute distribution statistics
                values = self.reference_data[column].dropna().values
                
                if len(values) == 0:
                    continue
                
                self.reference_distributions[column] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                    "q1": float(np.percentile(values, 25)),
                    "q3": float(np.percentile(values, 75)),
                    "histogram": np.histogram(values, bins=20, density=True)
                }
            except Exception as e:
                logger.error(f"Error computing distribution for {column}: {str(e)}")
    
    def record_prediction(
        self,
        prediction: float,
        label: Optional[int] = None,
        features: Optional[Dict] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a prediction for monitoring.
        
        Args:
            prediction: Predicted fraud probability
            label: Actual fraud label (0 for legitimate, 1 for fraud)
            features: Features used for prediction
            timestamp: Timestamp of the prediction
        """
        with self.lock:
            # Use current time if timestamp not provided
            if timestamp is None:
                timestamp = datetime.now()
            
            # Record prediction
            self.predictions.append(float(prediction))
            
            # Record label if provided
            if label is not None:
                self.labels.append(int(label))
            else:
                # Use None as placeholder
                self.labels.append(None)
            
            # Record timestamp
            self.timestamps.append(timestamp)
            
            # Check for drift if features provided
            if features is not None and self.reference_distributions:
                self._check_drift(features)
            
            # Update performance metrics if enough labeled data
            if len(self.labels) >= 100 and all(l is not None for l in self.labels):
                self._update_performance_metrics()
    
    def _check_drift(self, features: Dict) -> None:
        """
        Check for data drift in features.
        
        Args:
            features: Features to check for drift
        """
        drift_detected = False
        drift_features = {}
        
        for feature, value in features.items():
            if feature in self.reference_distributions:
                ref = self.reference_distributions[feature]
                
                # Skip non-numeric values
                if not isinstance(value, (int, float)):
                    continue
                
                # Check if value is outside expected range
                z_score = abs((value - ref["mean"]) / max(ref["std"], 1e-10))
                
                if z_score > 3:  # More than 3 standard deviations
                    drift_detected = True
                    drift_features[feature] = {
                        "value": value,
                        "z_score": z_score,
                        "reference_mean": ref["mean"],
                        "reference_std": ref["std"]
                    }
        
        if drift_detected:
            # Record drift
            timestamp = datetime.now().isoformat()
            
            if "feature_drift" not in self.metrics_history["drift"]:
                self.metrics_history["drift"]["feature_drift"] = {}
            
            for feature, details in drift_features.items():
                if feature not in self.metrics_history["drift"]["feature_drift"]:
                    self.metrics_history["drift"]["feature_drift"][feature] = []
                
                self.metrics_history["drift"]["feature_drift"][feature].append({
                    "timestamp": timestamp,
                    "details": details
                })
            
            # Save metrics
            self._save_metrics()
            
            # Log drift
            logger.warning(f"Data drift detected in features: {list(drift_features.keys())}")
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics based on recorded predictions and labels."""
        # Convert to numpy arrays
        y_true = np.array([l for l in self.labels if l is not None])
        y_pred_proba = np.array([self.predictions[i] for i, l in enumerate(self.labels) if l is not None])
        
        if len(y_true) < 2 or len(np.unique(y_true)) < 2:
            logger.warning("Not enough labeled data or all labels are the same")
            return
        
        try:
            # Calculate metrics
            auc_roc = roc_auc_score(y_true, y_pred_proba)
            average_precision = average_precision_score(y_true, y_pred_proba)
            
            # Convert probabilities to binary predictions
            threshold = 0.5
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            # Record metrics
            timestamp = datetime.now().isoformat()
            
            self.metrics_history["performance"]["auc_roc"].append(float(auc_roc))
            self.metrics_history["performance"]["precision"].append(float(precision))
            self.metrics_history["performance"]["recall"].append(float(recall))
            self.metrics_history["performance"]["f1_score"].append(float(f1))
            self.metrics_history["performance"]["average_precision"].append(float(average_precision))
            self.metrics_history["performance"]["timestamps"].append(timestamp)
            
            # Check for performance degradation
            self._check_performance_degradation()
            
            # Save metrics
            self._save_metrics()
            
            logger.info(f"Updated performance metrics: AUC={auc_roc:.4f}, F1={f1:.4f}")
        
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
    
    def _check_performance_degradation(self) -> None:
        """Check for performance degradation and generate alerts."""
        # Need at least 2 measurements to detect degradation
        if len(self.metrics_history["performance"]["auc_roc"]) < 2:
            return
        
        # Get current and previous metrics
        current_auc = self.metrics_history["performance"]["auc_roc"][-1]
        current_f1 = self.metrics_history["performance"]["f1_score"][-1]
        
        # Calculate baseline as average of previous metrics
        baseline_auc = np.mean(self.metrics_history["performance"]["auc_roc"][:-1])
        baseline_f1 = np.mean(self.metrics_history["performance"]["f1_score"][:-1])
        
        # Check for degradation
        auc_degradation = (baseline_auc - current_auc) / baseline_auc
        f1_degradation = (baseline_f1 - current_f1) / baseline_f1
        
        if auc_degradation > self.alert_threshold or f1_degradation > self.alert_threshold:
            # Generate alert
            alert = {
                "timestamp": datetime.now().isoformat(),
                "type": "performance_degradation",
                "details": {
                    "auc_degradation": float(auc_degradation),
                    "f1_degradation": float(f1_degradation),
                    "current_auc": float(current_auc),
                    "baseline_auc": float(baseline_auc),
                    "current_f1": float(current_f1),
                    "baseline_f1": float(baseline_f1)
                }
            }
            
            self.metrics_history["alerts"].append(alert)
            
            # Log alert
            logger.warning(
                f"Performance degradation detected: "
                f"AUC degradation={auc_degradation:.4f}, "
                f"F1 degradation={f1_degradation:.4f}"
            )
    
    def get_performance_metrics(self) -> Dict:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        with self.lock:
            # Convert to numpy arrays
            y_true = np.array([l for l in self.labels if l is not None])
            y_pred_proba = np.array([self.predictions[i] for i, l in enumerate(self.labels) if l is not None])
            
            if len(y_true) < 2 or len(np.unique(y_true)) < 2:
                return {
                    "status": "insufficient_data",
                    "message": "Not enough labeled data or all labels are the same"
                }
            
            try:
                # Calculate metrics
                auc_roc = roc_auc_score(y_true, y_pred_proba)
                average_precision = average_precision_score(y_true, y_pred_proba)
                
                # Convert probabilities to binary predictions
                threshold = 0.5
                y_pred = (y_pred_proba >= threshold).astype(int)
                
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                
                # Calculate confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel()
                
                return {
                    "status": "success",
                    "metrics": {
                        "auc_roc": float(auc_roc),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1_score": float(f1),
                        "average_precision": float(average_precision),
                        "confusion_matrix": {
                            "true_negative": int(tn),
                            "false_positive": int(fp),
                            "false_negative": int(fn),
                            "true_positive": int(tp)
                        },
                        "sample_size": len(y_true)
                    }
                }
            
            except Exception as e:
                return {
                    "status": "error",
                    "message": str(e)
                }
    
    def get_drift_metrics(self) -> Dict:
        """
        Get current drift metrics.
        
        Returns:
            Dictionary of drift metrics
        """
        with self.lock:
            return {
                "status": "success",
                "metrics": {
                    "feature_drift": self.metrics_history["drift"]["feature_drift"],
                    "concept_drift": self.metrics_history["drift"]["concept_drift"]
                }
            }
    
    def get_alerts(self) -> Dict:
        """
        Get current alerts.
        
        Returns:
            Dictionary of alerts
        """
        with self.lock:
            return {
                "status": "success",
                "alerts": self.metrics_history["alerts"]
            }
    
    def plot_performance_metrics(self, save_path: Optional[str] = None) -> None:
        """
        Plot performance metrics over time.
        
        Args:
            save_path: Path to save the plot (if None, display the plot)
        """
        with self.lock:
            if len(self.metrics_history["performance"]["timestamps"]) == 0:
                logger.warning("No performance metrics to plot")
                return
            
            # Convert timestamps to datetime objects
            timestamps = [datetime.fromisoformat(ts) for ts in self.metrics_history["performance"]["timestamps"]]
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot metrics
            plt.subplot(2, 1, 1)
            plt.plot(timestamps, self.metrics_history["performance"]["auc_roc"], label="AUC-ROC")
            plt.plot(timestamps, self.metrics_history["performance"]["average_precision"], label="Average Precision")
            plt.title("Model Performance Metrics Over Time")
            plt.ylabel("Score")
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.plot(timestamps, self.metrics_history["performance"]["precision"], label="Precision")
            plt.plot(timestamps, self.metrics_history["performance"]["recall"], label="Recall")
            plt.plot(timestamps, self.metrics_history["performance"]["f1_score"], label="F1 Score")
            plt.xlabel("Time")
            plt.ylabel("Score")
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            
            # Save or display the plot
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Performance metrics plot saved to {save_path}")
            else:
                plt.show()
    
    def save_reference_data(self, path: str) -> None:
        """
        Save reference data and distributions for drift detection.
        
        Args:
            path: Path to save the reference data
        """
        with self.lock:
            if self.reference_data is None:
                logger.warning("No reference data to save")
                return
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save reference data
            self.reference_data.to_csv(f"{path}_data.csv", index=False)
            
            # Save reference distributions
            with open(f"{path}_distributions.pkl", 'wb') as f:
                pickle.dump(self.reference_distributions, f)
            
            logger.info(f"Reference data and distributions saved to {path}")
    
    def load_reference_data(self, path: str) -> None:
        """
        Load reference data and distributions for drift detection.
        
        Args:
            path: Path to load the reference data from
        """
        with self.lock:
            # Load reference data
            data_path = f"{path}_data.csv"
            if os.path.exists(data_path):
                self.reference_data = pd.read_csv(data_path)
                logger.info(f"Loaded reference data from {data_path}")
            else:
                logger.warning(f"Reference data file not found: {data_path}")
            
            # Load reference distributions
            dist_path = f"{path}_distributions.pkl"
            if os.path.exists(dist_path):
                with open(dist_path, 'rb') as f:
                    self.reference_distributions = pickle.load(f)
                logger.info(f"Loaded reference distributions from {dist_path}")
            else:
                logger.warning(f"Reference distributions file not found: {dist_path}")
                
                # Compute distributions if reference data is available
                if self.reference_data is not None:
                    self._compute_reference_distributions()


class ModelMonitoringService:
    """
    Service for model monitoring in production.
    
    This class provides a background service for monitoring model performance
    and data drift in production, with periodic reporting and alerting.
    """
    
    def __init__(
        self,
        model_monitor: ModelMonitor,
        reporting_interval: int = 3600,  # 1 hour
        alert_webhook: Optional[str] = None
    ):
        """
        Initialize the monitoring service.
        
        Args:
            model_monitor: ModelMonitor instance
            reporting_interval: Interval for periodic reporting in seconds
            alert_webhook: Webhook URL for sending alerts
        """
        self.model_monitor = model_monitor
        self.reporting_interval = reporting_interval
        self.alert_webhook = alert_webhook
        
        self.running = False
        self.thread = None
        
        logger.info("Initialized ModelMonitoringService")
    
    def start(self) -> None:
        """Start the monitoring service."""
        if self.running:
            logger.warning("Monitoring service is already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("Started model monitoring service")
    
    def stop(self) -> None:
        """Stop the monitoring service."""
        if not self.running:
            logger.warning("Monitoring service is not running")
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        logger.info("Stopped model monitoring service")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        last_report_time = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Generate periodic report
                if current_time - last_report_time >= self.reporting_interval:
                    self._generate_report()
                    last_report_time = current_time
                
                # Check for alerts
                self._check_alerts()
                
                # Sleep for a short time
                time.sleep(10)
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)  # Sleep longer on error
    
    def _generate_report(self) -> None:
        """Generate periodic monitoring report."""
        try:
            # Get current metrics
            performance_metrics = self.model_monitor.get_performance_metrics()
            drift_metrics = self.model_monitor.get_drift_metrics()
            
            # Generate report
            report = {
                "timestamp": datetime.now().isoformat(),
                "performance": performance_metrics,
                "drift": drift_metrics
            }
            
            # Save report
            report_file = f"monitoring/reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Generate performance plot
            plot_file = f"monitoring/reports/performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.model_monitor.plot_performance_metrics(save_path=plot_file)
            
            logger.info(f"Generated monitoring report: {report_file}")
        
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
    
    def _check_alerts(self) -> None:
        """Check for alerts and send notifications."""
        try:
            # Get alerts
            alerts = self.model_monitor.get_alerts()
            
            # Check for new alerts
            if alerts["status"] == "success" and alerts["alerts"]:
                # Get the latest alert
                latest_alert = alerts["alerts"][-1]
                
                # Check if alert is new (within the last minute)
                alert_time = datetime.fromisoformat(latest_alert["timestamp"])
                if (datetime.now() - alert_time).total_seconds() < 60:
                    self._send_alert(latest_alert)
        
        except Exception as e:
            logger.error(f"Error checking alerts: {str(e)}")
    
    def _send_alert(self, alert: Dict) -> None:
        """
        Send alert notification.
        
        Args:
            alert: Alert details
        """
        if not self.alert_webhook:
            logger.warning("No alert webhook configured")
            return
        
        try:
            import requests
            
            # Format alert message
            if alert["type"] == "performance_degradation":
                message = (
                    f"⚠️ **Performance Degradation Detected**\n\n"
                    f"- AUC degradation: {alert['details']['auc_degradation']:.2%}\n"
                    f"- F1 degradation: {alert['details']['f1_degradation']:.2%}\n"
                    f"- Current AUC: {alert['details']['current_auc']:.4f}\n"
                    f"- Baseline AUC: {alert['details']['baseline_auc']:.4f}\n\n"
                    f"Time: {alert['timestamp']}"
                )
            else:
                message = f"⚠️ **Alert: {alert['type']}**\n\nTime: {alert['timestamp']}"
            
            # Send webhook
            payload = {
                "text": message
            }
            
            response = requests.post(
                self.alert_webhook,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                logger.info(f"Alert sent successfully: {alert['type']}")
            else:
                logger.error(f"Failed to send alert: {response.status_code} {response.text}")
        
        except Exception as e:
            logger.error(f"Error sending alert: {str(e)}")


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.99, 0.01],  # Imbalanced dataset
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    data = pd.DataFrame(X, columns=feature_names)
    data["is_fraud"] = y
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("is_fraud", axis=1),
        data["is_fraud"],
        test_size=0.2,
        random_state=42,
        stratify=data["is_fraud"]
    )
    
    # Initialize model monitor with reference data
    monitor = ModelMonitor(
        metrics_file="monitoring/metrics.json",
        reference_data=X_train
    )
    
    # Simulate predictions
    import random
    
    # Good predictions
    for i in range(500):
        # Simulate good model (AUC ~0.9)
        if y_test.iloc[i] == 1:
            pred = random.uniform(0.7, 1.0)
        else:
            pred = random.uniform(0.0, 0.3)
        
        monitor.record_prediction(
            prediction=pred,
            label=y_test.iloc[i],
            features=X_test.iloc[i].to_dict()
        )
    
    # Get performance metrics
    print("Performance metrics:")
    print(monitor.get_performance_metrics())
    
    # Plot performance metrics
    monitor.plot_performance_metrics()
    
    # Save reference data
    monitor.save_reference_data("monitoring/reference")
    
    # Initialize monitoring service
    service = ModelMonitoringService(
        model_monitor=monitor,
        reporting_interval=60,  # 1 minute for demonstration
        alert_webhook=None  # No webhook for demonstration
    )
    
    # Start monitoring service
    service.start()
    
    # Simulate degraded predictions
    for i in range(500, 1000):
        # Simulate degraded model (AUC ~0.7)
        if y_test.iloc[i] == 1:
            pred = random.uniform(0.5, 0.9)
        else:
            pred = random.uniform(0.1, 0.5)
        
        monitor.record_prediction(
            prediction=pred,
            label=y_test.iloc[i],
            features=X_test.iloc[i].to_dict()
        )
    
    # Wait for monitoring service to generate report
    time.sleep(65)
    
    # Stop monitoring service
    service.stop()
    
    # Get final performance metrics
    print("\nFinal performance metrics:")
    print(monitor.get_performance_metrics())
    
    # Get alerts
    print("\nAlerts:")
    print(monitor.get_alerts())

