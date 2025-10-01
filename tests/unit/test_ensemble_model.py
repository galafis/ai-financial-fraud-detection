import unittest
import pandas as pd
import numpy as np
import os
import pickle
from unittest.mock import MagicMock, patch

# Adiciona o diretório 'src' ao PYTHONPATH para que as importações funcionem
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from models.ensemble_model import FraudDetectionEnsemble

class TestFraudDetectionEnsemble(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.models_dir = "./test_models"
        os.makedirs(cls.models_dir, exist_ok=True)

        # Criar modelos dummy reais que podem ser serializados
        cls.create_real_dummy_models()

    @classmethod
    def tearDownClass(cls):
        # Limpa os arquivos de modelo de teste
        for f in os.listdir(cls.models_dir):
            os.remove(os.path.join(cls.models_dir, f))
        os.rmdir(cls.models_dir)

    def setUp(self):
        self.X_test = pd.DataFrame(np.random.rand(10, 5), columns=[f'feature_{i}' for i in range(5)])
        self.ensemble_model = FraudDetectionEnsemble.load(self.models_dir)

    def test_predict_proba(self):
        probabilities = self.ensemble_model.predict_proba(self.X_test)
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(probabilities.shape, (len(self.X_test),))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))

    def test_predict(self):
        predictions = self.ensemble_model.predict(self.X_test)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape, (len(self.X_test),))
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))

    @staticmethod
    def create_real_dummy_models():
        model_base_path = TestFraudDetectionEnsemble.models_dir

        config = {
            'rf_params': {}, 'xgb_params': {}, 'nn_params': {'input_dim': 5}, 'ae_params': {'input_dim': 5}, 'meta_params': {},
            'use_calibration': False, 'threshold': 0.5, 'random_state': 42
        }
        with open(os.path.join(model_base_path, 'config.pkl'), 'wb') as f:
            pickle.dump(config, f)

        X_dummy = np.random.rand(10, 5)
        y_dummy = np.random.randint(0, 2, 10)

        from sklearn.ensemble import RandomForestClassifier
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_dummy, y_dummy)
        with open(os.path.join(model_base_path, 'random_forest.pkl'), 'wb') as f:
            pickle.dump(rf_model, f)

        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(random_state=42)
        xgb_model.fit(X_dummy, y_dummy)
        with open(os.path.join(model_base_path, 'xgboost.pkl'), 'wb') as f:
            pickle.dump(xgb_model, f)

        from tensorflow.keras.models import Sequential, Model
        from tensorflow.keras.layers import Dense, Input
        from tensorflow.keras.optimizers import Adam

        nn_model = Sequential([
            Dense(10, activation='relu', input_shape=(5,)),
            Dense(1, activation='sigmoid')
        ])
        nn_model.compile(optimizer=Adam(), loss='binary_crossentropy')
        nn_model.fit(X_dummy, y_dummy, epochs=1, verbose=0)
        nn_model.save(os.path.join(model_base_path, 'neural_network.h5'))

        input_layer = Input(shape=(5,))
        encoded = Dense(3, activation='relu')(input_layer)
        decoded = Dense(5, activation='linear')(encoded)
        ae_model = Model(inputs=input_layer, outputs=decoded)
        ae_model.compile(optimizer=Adam(), loss='mean_squared_error')
        ae_model.fit(X_dummy, X_dummy, epochs=1, verbose=0)
        ae_model.save(os.path.join(model_base_path, 'autoencoder.h5'))

        X_meta_dummy = np.random.rand(10, 4)
        meta_model = RandomForestClassifier(random_state=42)
        meta_model.fit(X_meta_dummy, y_dummy)
        with open(os.path.join(model_base_path, 'meta_model.pkl'), 'wb') as f:
            pickle.dump(meta_model, f)

        from sklearn.preprocessing import StandardScaler
        feature_scaler = StandardScaler()
        feature_scaler.fit(X_dummy)
        with open(os.path.join(model_base_path, 'feature_scaler.pkl'), 'wb') as f:
            pickle.dump(feature_scaler, f)

        ae_scaler = StandardScaler()
        ae_scaler.fit(X_dummy)
        with open(os.path.join(model_base_path, 'ae_scaler.pkl'), 'wb') as f:
            pickle.dump(ae_scaler, f)

if __name__ == '__main__':
    unittest.main()

