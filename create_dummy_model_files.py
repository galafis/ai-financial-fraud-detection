
import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define o caminho base para salvar os modelos
model_base_path = 'ai-financial-fraud-detection/models/ensemble_model'
os.makedirs(model_base_path, exist_ok=True)

# 1. Criar config.pkl
config = {
    'rf_params': {}, 'xgb_params': {}, 'nn_params': {'input_dim': 2}, 'ae_params': {'input_dim': 2}, 'meta_params': {},
    'use_calibration': True, 'threshold': 0.5, 'random_state': 42
}
with open(os.path.join(model_base_path, 'config.pkl'), 'wb') as f:
    pickle.dump(config, f)
print(f'Created {os.path.join(model_base_path, "config.pkl")}')

# 2. Criar random_forest.pkl
rf_model = RandomForestClassifier(random_state=42)
# Fit with dummy data to make it a valid estimator
rf_model.fit(np.random.rand(10, 2), np.random.randint(0, 2, 10))
with open(os.path.join(model_base_path, 'random_forest.pkl'), 'wb') as f:
    pickle.dump(rf_model, f)
print(f'Created {os.path.join(model_base_path, "random_forest.pkl")}')

# 3. Criar xgboost.pkl
xgb_model = xgb.XGBClassifier(random_state=42)
# Fit with dummy data
xgb_model.fit(np.random.rand(10, 2), np.random.randint(0, 2, 10))
with open(os.path.join(model_base_path, 'xgboost.pkl'), 'wb') as f:
    pickle.dump(xgb_model, f)
print(f'Created {os.path.join(model_base_path, "xgboost.pkl")}')

# 4. Criar neural_network.h5
nn_model = Sequential([
    Dense(10, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])
nn_model.compile(optimizer='adam', loss='binary_crossentropy')
nn_model.save(os.path.join(model_base_path, 'neural_network.h5'))
print(f'Created {os.path.join(model_base_path, "neural_network.h5")}')

# 5. Criar autoencoder.h5
ae_model = Sequential([
    Dense(5, activation='relu', input_shape=(2,)),
    Dense(2, activation='linear') # Output dimension should match input dimension
])
ae_model.compile(optimizer='adam', loss='mean_squared_error')
ae_model.save(os.path.join(model_base_path, 'autoencoder.h5'))
print(f'Created {os.path.join(model_base_path, "autoencoder.h5")}')

# 6. Criar meta_model.pkl
meta_model = RandomForestClassifier(random_state=42) # Usando RF como meta-modelo dummy
meta_model.fit(np.random.rand(10, 4), np.random.randint(0, 2, 10)) # 4 features para meta-modelo (rf_proba, xgb_proba, nn_proba, ae_scores)
with open(os.path.join(model_base_path, 'meta_model.pkl'), 'wb') as f:
    pickle.dump(meta_model, f)
print(f'Created {os.path.join(model_base_path, "meta_model.pkl")}')

# 7. Criar feature_scaler.pkl
feature_scaler = StandardScaler()
feature_scaler.fit(np.random.rand(10, 2)) # Fit com 2 features
with open(os.path.join(model_base_path, 'feature_scaler.pkl'), 'wb') as f:
    pickle.dump(feature_scaler, f)
print(f'Created {os.path.join(model_base_path, "feature_scaler.pkl")}')

# 8. Criar ae_scaler.pkl
ae_scaler = StandardScaler()
ae_scaler.fit(np.random.rand(10, 2)) # Fit com 2 features
with open(os.path.join(model_base_path, 'ae_scaler.pkl'), 'wb') as f:
    pickle.dump(ae_scaler, f)
print(f'Created {os.path.join(model_base_path, "ae_scaler.pkl")}')

# Calibrators (opcionais, criar apenas se use_calibration for True)
# Para simplificar, não criaremos calibradores dummy agora, pois o erro não é sobre eles.
# Se o backtest.py falhar por falta de calibradores, adicionaremos aqui.

print("All dummy model files created successfully.")

