"""
AI-Powered Financial Fraud Detection System
Logger Utility

This module provides logging functionality for the fraud detection system.

Author: Gabriel Demetrios Lafis
Date: June 2025
"""

import os
import logging
import logging.handlers
from datetime import datetime
from typing import Optional

from ..config.api_config import LOG_LEVEL, LOG_FORMAT, LOG_FILE

# Create logs directory if it doesn't exist
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Configure root logger
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),  # Console handler
        logging.handlers.RotatingFileHandler(
            LOG_FILE,
            maxBytes=10485760,  # 10MB
            backupCount=10,
            encoding="utf-8"
        )
    ]
)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Name of the logger
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TransactionLogger:
    """
    Logger for transaction data and fraud predictions.
    
    This class provides methods to log transaction details and fraud predictions
    to a separate log file for audit and compliance purposes.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize the transaction logger.
        
        Args:
            log_file: Path to the log file (defaults to logs/transactions.log)
        """
        if log_file is None:
            log_file = "logs/transactions.log"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger("transaction_logger")
        self.logger.setLevel(logging.INFO)
        
        # Avoid adding duplicate handlers on repeated imports
        if not self.logger.handlers:
            # Create file handler
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=104857600,  # 100MB
                backupCount=20,
                encoding="utf-8"
            )
            
            # Create formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            
            # Add formatter to handler
            handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(handler)
    
    def log_transaction(
        self,
        transaction_id: str,
        amount: float,
        merchant_id: str,
        customer_id: str,
        timestamp: Optional[datetime] = None,
        fraud_probability: Optional[float] = None,
        is_fraud: Optional[bool] = None,
        additional_data: Optional[dict] = None
    ) -> None:
        """
        Log a transaction and its fraud prediction.
        
        Args:
            transaction_id: Unique identifier for the transaction
            amount: Transaction amount
            merchant_id: Identifier for the merchant
            customer_id: Identifier for the customer
            timestamp: Transaction timestamp (defaults to current time)
            fraud_probability: Predicted fraud probability (if available)
            is_fraud: Binary fraud prediction (if available)
            additional_data: Additional data to log
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Create log message
        log_data = {
            "transaction_id": transaction_id,
            "amount": amount,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "timestamp": timestamp.isoformat(),
        }
        
        # Add fraud prediction if available
        if fraud_probability is not None:
            log_data["fraud_probability"] = fraud_probability
        
        if is_fraud is not None:
            log_data["is_fraud"] = is_fraud
        
        # Add additional data if provided
        if additional_data:
            log_data.update(additional_data)
        
        # Log as JSON string
        import json
        self.logger.info(json.dumps(log_data))


# Create a global transaction logger instance
transaction_logger = TransactionLogger()


if __name__ == "__main__":
    # Example usage
    logger = get_logger(__name__)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Example transaction logging
    transaction_logger.log_transaction(
        transaction_id="TX123456",
        amount=1000.00,
        merchant_id="MERCH001",
        customer_id="CUST789",
        fraud_probability=0.95,
        is_fraud=True,
        additional_data={"payment_method": "credit_card", "ip_address": "192.168.1.1"}
    )

