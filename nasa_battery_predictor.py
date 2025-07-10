#!/usr/bin/env python3
"""
NASA Battery Dataset - Remaining Useful Life and State of Health Prediction

This module replicates the functionality from the NASA Battery project:
1. Predicting remaining capacity (SoH) of Li-ion batteries
2. Predicting future discharge curves using LSTM

Original project reference: https://github.com/natskiu/Nasa-Battery
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import lightgbm as lgb

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import joblib
import os
from datetime import datetime

class NASABatteryDataLoader:
    """Data loader for NASA Battery dataset (.mat files)"""
    
    def __init__(self):
        self.battery_names = ['B0005', 'B0006', 'B0007', 'B0018', 'B0025', 'B0026', 'B0027', 'B0028']
        self.data = {}
        
    def load_mat_file(self, file_path):
        """Load NASA battery .mat file"""
        try:
            mat_data = loadmat(file_path)
            battery_name = os.path.basename(file_path).split('.')[0]
            
            # Extract the battery data structure
            battery_key = [key for key in mat_data.keys() if key.startswith('B')][0]
            battery_data = mat_data[battery_key]
            
            cycles = []
            for i in range(len(battery_data['cycle'][0, 0][0])):
                cycle_data = {}
                
                # Extract cycle information
                cycle_info = battery_data['cycle'][0, 0][0, i]
                
                # Get type (charge, discharge, impedance)
                cycle_type = str(cycle_info['type'][0, 0][0])
                
                # Get data based on type
                if cycle_type == 'discharge':
                    data = cycle_info['data'][0, 0]
                    
                    cycle_data['type'] = cycle_type
                    cycle_data['ambient_temperature'] = float(cycle_info['ambient_temperature'][0, 0][0, 0])
                    cycle_data['time'] = data['Time'][0, 0].flatten()
                    cycle_data['voltage'] = data['Voltage_measured'][0, 0].flatten()
                    cycle_data['current'] = data['Current_measured'][0, 0].flatten()
                    cycle_data['temperature'] = data['Temperature_measured'][0, 0].flatten()
                    cycle_data['capacity'] = data['Capacity'][0, 0].flatten()[-1] if 'Capacity' in data.dtype.names else None
                    
                    cycles.append(cycle_data)
            
            print(f"Loaded {len(cycles)} discharge cycles from {battery_name}")
            return cycles
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def load_multiple_batteries(self, data_directory):
        """Load multiple battery .mat files from directory"""
        battery_data = {}
        
        for battery_name in self.battery_names:
            file_path = os.path.join(data_directory, f"{battery_name}.mat")
            if os.path.exists(file_path):
                cycles = self.load_mat_file(file_path)
                if cycles:
                    battery_data[battery_name] = cycles
                    
        return battery_data
    
    def create_sample_data(self):
        """Create sample NASA battery data for demonstration"""
        print("Creating sample NASA battery data...")
        
        sample_data = {}
        battery_names = ['B0005', 'B0006', 'B0007', 'B0018']
        
        for battery_name in battery_names:
            cycles = []
            n_cycles = np.random.randint(120, 170)  # Different cycle counts for each battery
            
            # Initial capacity
            initial_capacity = 2.0 + np.random.normal(0, 0.05)
            
            for cycle_idx in range(n_cycles):
                # Simulate capacity degradation
                degradation_rate = 0.0008 + np.random.normal(0, 0.0001)
                capacity = initial_capacity * (1 - degradation_rate * cycle_idx)
                capacity += np.random.normal(0, 0.01)  # Add noise
                capacity = max(capacity, 1.0)  # Minimum capacity
                
                # Simulate discharge curve
                n_points = np.random.randint(800, 1200)
                time = np.linspace(0, capacity * 3600 / 2, n_points)  # Time in seconds
                
                # Voltage curve (decreasing during discharge)
                voltage = 4.2 - (4.2 - 2.5) * (time / time[-1]) ** 0.8
                voltage += np.random.normal(0, 0.02, len(voltage))  # Add noise
                
                # Current (approximately constant during discharge)
                current = 2.0 + np.random.normal(0, 0.1, len(voltage))
                
                # Temperature (increases during discharge)
                temp_rise = 5 * (time / time[-1]) ** 0.5
                temperature = 24 + temp_rise + np.random.normal(0, 0.5, len(voltage))
                
                cycle_data = {
                    'type': 'discharge',
                    'ambient_temperature': 24.0,
                    'time': time,
                    'voltage': voltage,
                    'current': current,
                    'temperature': temperature,
                    'capacity': capacity
                }
                
                cycles.append(cycle_data)
            
            sample_data[battery_name] = cycles
            print(f"Created {len(cycles)} cycles for {battery_name}")
        
        return sample_data

class FeatureExtractor:
    """Extract features from NASA battery discharge cycles"""
    
    def __init__(self):
        self.feature_names = [
            'time_to_max_temp',
            'max_temperature',
            'avg_temp_rise_rate',
            'time_to_3V',
            'initial_voltage_slope',
            'capacity',
            'cycle_number',
            'ambient_temperature'
        ]
    
    def extract_features_from_cycle(self, cycle_data, cycle_number):
        """Extract features from a single discharge cycle"""
        try:
            time = cycle_data['time']
            voltage = cycle_data['voltage']
            temperature = cycle_data['temperature']
            capacity = cycle_data['capacity']
            ambient_temp = cycle_data['ambient_temperature']
            
            features = {}
            
            # 1. Time taken for temperature to reach maximum
            max_temp_idx = np.argmax(temperature)
            features['time_to_max_temp'] = time[max_temp_idx] if max_temp_idx > 0 else 0
            
            # 2. Maximum temperature reached
            features['max_temperature'] = np.max(temperature)
            
            # 3. Average rate of temperature increase
            temp_rise = features['max_temperature'] - temperature[0]
            time_to_max = features['time_to_max_temp']
            features['avg_temp_rise_rate'] = temp_rise / time_to_max if time_to_max > 0 else 0
            
            # 4. Time for voltage to drop below 3V
            below_3v_indices = np.where(voltage < 3.0)[0]
            features['time_to_3V'] = time[below_3v_indices[0]] if len(below_3v_indices) > 0 else time[-1]
            
            # 5. Initial slope of voltage
            if len(voltage) > 10:
                initial_points = min(10, len(voltage) // 10)
                voltage_slope = np.polyfit(time[:initial_points], voltage[:initial_points], 1)[0]
                features['initial_voltage_slope'] = voltage_slope
            else:
                features['initial_voltage_slope'] = 0
            
            # 6. Capacity (target variable)
            features['capacity'] = capacity
            
            # 7. Cycle number
            features['cycle_number'] = cycle_number
            
            # 8. Ambient temperature
            features['ambient_temperature'] = ambient_temp
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def extract_features_from_battery(self, battery_cycles):
        """Extract features from all cycles of a battery"""
        features_list = []
        
        for cycle_idx, cycle_data in enumerate(battery_cycles):
            if cycle_data['type'] == 'discharge':
                features = self.extract_features_from_cycle(cycle_data, cycle_idx + 1)
                if features:
                    features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def extract_features_from_dataset(self, battery_dataset):
        """Extract features from entire battery dataset"""
        all_features = []
        
        for battery_name, cycles in battery_dataset.items():
            print(f"Extracting features from {battery_name}...")
            battery_features = self.extract_features_from_battery(cycles)
            battery_features['battery_name'] = battery_name
            all_features.append(battery_features)
        
        combined_features = pd.concat(all_features, ignore_index=True)
        print(f"Total features extracted: {len(combined_features)} samples")
        
        return combined_features

class SoHPredictor:
    """State of Health Prediction using ensemble methods"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.ensemble_model = None
        self.feature_importance = None
        
    def prepare_data(self, features_df):
        """Prepare data for training"""
        # Remove outliers using Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_mask = iso_forest.fit_predict(features_df.select_dtypes(include=[np.number])) == 1
        
        clean_data = features_df[outlier_mask].copy()
        print(f"Removed {len(features_df) - len(clean_data)} outliers")
        
        # Prepare features and target
        feature_cols = ['time_to_max_temp', 'max_temperature', 'avg_temp_rise_rate', 
                       'time_to_3V', 'initial_voltage_slope', 'cycle_number', 'ambient_temperature']
        
        X = clean_data[feature_cols].fillna(0)
        y = clean_data['capacity']
        
        return X, y, clean_data
    
    def train_baseline_models(self, X_train, y_train, X_val, y_val):
        """Train baseline models"""
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1),
            'Linear Regression': LinearRegression(),
            'Elastic Net': ElasticNet(random_state=42),
            'SVR': SVR(kernel='rbf'),
            'KNN': KNeighborsRegressor(n_neighbors=5)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            try:
                # Cross-validation on training set
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                          scoring='neg_mean_squared_error')
                cv_rmse = np.sqrt(-cv_scores.mean())
                
                # Train on full training set
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                y_pred = model.predict(X_val)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                val_r2 = r2_score(y_val, y_pred)
                
                results[name] = {
                    'model': model,
                    'cv_rmse': cv_rmse,
                    'val_rmse': val_rmse,
                    'val_r2': val_r2,
                    'predictions': y_pred
                }
                
                print(f"  CV RMSE: {cv_rmse:.6f}, Val RMSE: {val_rmse:.6f}, Val R²: {val_r2:.6f}")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
        
        self.models = results
        return results
    
    def tune_best_models(self, X_train, y_train, X_val, y_val):
        """Hyperparameter tuning for best models"""
        # Select top 3 models based on validation RMSE
        sorted_models = sorted(self.models.items(), key=lambda x: x[1]['val_rmse'])
        best_models = dict(sorted_models[:3])
        
        print(f"Tuning hyperparameters for: {list(best_models.keys())}")
        
        tuned_models = {}
        
        for name, model_info in best_models.items():
            print(f"Tuning {name}...")
            
            if name == 'Random Forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
                base_model = RandomForestRegressor(random_state=42)
                
            elif name == 'Extra Trees':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
                base_model = ExtraTreesRegressor(random_state=42)
                
            elif name == 'XGBoost':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
                base_model = xgb.XGBRegressor(random_state=42, verbosity=0)
                
            else:
                # Use original model if no specific tuning
                tuned_models[name] = model_info['model']
                continue
            
            try:
                grid_search = GridSearchCV(base_model, param_grid, cv=3, 
                                         scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                
                # Evaluate tuned model
                y_pred = grid_search.best_estimator_.predict(X_val)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                val_r2 = r2_score(y_val, y_pred)
                
                tuned_models[name] = grid_search.best_estimator_
                
                print(f"  Best params: {grid_search.best_params_}")
                print(f"  Tuned Val RMSE: {val_rmse:.6f}, Val R²: {val_r2:.6f}")
                
            except Exception as e:
                print(f"  Error tuning {name}: {e}")
                tuned_models[name] = model_info['model']
        
        return tuned_models
    
    def create_ensemble(self, tuned_models, X_val, y_val):
        """Create weighted voting ensemble"""
        # Calculate weights based on validation performance
        weights = []
        estimators = []
        
        for name, model in tuned_models.items():
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            weight = 1.0 / rmse  # Inverse of RMSE as weight
            
            weights.append(weight)
            estimators.append((name.replace(' ', '_').lower(), model))
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Create voting regressor
        voting_regressor = VotingRegressor(estimators, weights=weights)
        
        return voting_regressor, weights
    
    def train(self, features_df):
        """Complete training pipeline"""
        print("Preparing data...")
        X, y, clean_data = self.prepare_data(features_df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Stratified split by battery to ensure each battery is represented
        battery_names = clean_data['battery_name'].unique()
        train_batteries = battery_names[:int(0.6 * len(battery_names))]
        val_batteries = battery_names[int(0.6 * len(battery_names)):int(0.8 * len(battery_names))]
        test_batteries = battery_names[int(0.8 * len(battery_names)):]
        
        train_mask = clean_data['battery_name'].isin(train_batteries)
        val_mask = clean_data['battery_name'].isin(val_batteries)
        test_mask = clean_data['battery_name'].isin(test_batteries)
        
        X_train, y_train = X_scaled[train_mask], y.iloc[train_mask]
        X_val, y_val = X_scaled[val_mask], y.iloc[val_mask]
        X_test, y_test = X_scaled[test_mask], y.iloc[test_mask]
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Train baseline models
        print("\nTraining baseline models...")
        baseline_results = self.train_baseline_models(X_train, y_train, X_val, y_val)
        
        # Tune best models
        print("\nTuning hyperparameters...")
        tuned_models = self.tune_best_models(X_train, y_train, X_val, y_val)
        
        # Create ensemble
        print("\nCreating ensemble...")
        ensemble_model, weights = self.create_ensemble(tuned_models, X_val, y_val)
        ensemble_model.fit(X_train, y_train)
        
        # Final evaluation on test set
        y_pred_test = ensemble_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"\nFinal Ensemble Performance:")
        print(f"Test RMSE: {test_rmse:.6f}")
        print(f"Test R²: {test_r2:.6f}")
        
        self.ensemble_model = ensemble_model
        
        # Store feature importance
        if hasattr(list(tuned_models.values())[0], 'feature_importances_'):
            self.feature_importance = list(tuned_models.values())[0].feature_importances_
        
        return {
            'ensemble_model': ensemble_model,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'weights': weights,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_test': y_pred_test
        }
    
    def predict(self, features):
        """Make predictions using trained ensemble"""
        if self.ensemble_model is None:
            raise ValueError("Model not trained yet!")
        
        features_scaled = self.scaler.transform(features)
        return self.ensemble_model.predict(features_scaled)
    
    def save_model(self, filepath):
        """Save trained model"""
        model_data = {
            'ensemble_model': self.ensemble_model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.ensemble_model = model_data['ensemble_model']
        self.scaler = model_data['scaler']
        self.feature_importance = model_data.get('feature_importance')
        print(f"Model loaded from {filepath}")

class DischargeCurvePredictor:
    """LSTM-based discharge curve prediction"""
    
    def __init__(self, sequence_length=10, prediction_length=50):
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.scaler = MinMaxScaler()
        self.model = None
        
    def prepare_sequences(self, battery_dataset):
        """Prepare sequences for LSTM training"""
        all_sequences = []
        
        for battery_name, cycles in battery_dataset.items():
            print(f"Processing {battery_name}...")
            
            # Extract voltage curves from discharge cycles
            voltage_curves = []
            for cycle in cycles:
                if cycle['type'] == 'discharge' and len(cycle['voltage']) > 0:
                    # Normalize voltage curve to fixed length
                    voltage = cycle['voltage']
                    # Interpolate to fixed length (e.g., 100 points)
                    normalized_voltage = np.interp(
                        np.linspace(0, 1, 100),
                        np.linspace(0, 1, len(voltage)),
                        voltage
                    )
                    voltage_curves.append(normalized_voltage)
            
            # Create sequences
            for i in range(len(voltage_curves) - self.sequence_length - self.prediction_length):
                input_seq = voltage_curves[i:i + self.sequence_length]
                target_seq = voltage_curves[i + self.sequence_length:i + self.sequence_length + self.prediction_length]
                
                all_sequences.append((np.array(input_seq), np.array(target_seq)))
        
        return all_sequences
    
    def build_model(self, input_shape):
        """Build sequence-to-sequence LSTM model"""
        # Encoder
        encoder_inputs = Input(shape=input_shape)
        encoder_lstm = LSTM(64, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = Input(shape=(self.prediction_length, input_shape[1]))
        decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(input_shape[1], activation='linear')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # Model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return model
    
    def train(self, battery_dataset, epochs=100, batch_size=32):
        """Train LSTM model"""
        print("Preparing sequences...")
        sequences = self.prepare_sequences(battery_dataset)
        
        if len(sequences) == 0:
            print("No sequences prepared!")
            return None
        
        # Separate inputs and targets
        X_encoder = np.array([seq[0] for seq in sequences])
        y = np.array([seq[1] for seq in sequences])
        
        # Create decoder inputs (shifted target sequences)
        X_decoder = np.zeros_like(y)
        X_decoder[:, 1:, :] = y[:, :-1, :]
        
        print(f"Training on {len(sequences)} sequences")
        print(f"Encoder input shape: {X_encoder.shape}")
        print(f"Decoder input shape: {X_decoder.shape}")
        print(f"Target shape: {y.shape}")
        
        # Split data
        split_idx = int(0.8 * len(sequences))
        X_enc_train, X_enc_val = X_encoder[:split_idx], X_encoder[split_idx:]
        X_dec_train, X_dec_val = X_decoder[:split_idx], X_decoder[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build model
        self.model = self.build_model((self.sequence_length, X_encoder.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        # Train model
        history = self.model.fit(
            [X_enc_train, X_dec_train], y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([X_enc_val, X_dec_val], y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict_curves(self, input_sequences):
        """Predict future discharge curves"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare decoder input
        decoder_input = np.zeros((len(input_sequences), self.prediction_length, input_sequences.shape[2]))
        
        predictions = self.model.predict([input_sequences, decoder_input])
        return predictions
    
    def save_model(self, filepath):
        """Save LSTM model"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"LSTM model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load LSTM model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"LSTM model loaded from {filepath}")

class NASABatteryAnalyzer:
    """Main analyzer class combining SoH and discharge curve prediction"""
    
    def __init__(self):
        self.data_loader = NASABatteryDataLoader()
        self.feature_extractor = FeatureExtractor()
        self.soh_predictor = SoHPredictor()
        self.curve_predictor = DischargeCurvePredictor()
        self.battery_data = None
        self.features_df = None
        
    def load_data(self, data_directory=None, use_sample_data=True):
        """Load battery data"""
        if use_sample_data or data_directory is None:
            print("Using sample data...")
            self.battery_data = self.data_loader.create_sample_data()
        else:
            print(f"Loading data from {data_directory}...")
            self.battery_data = self.data_loader.load_multiple_batteries(data_directory)
        
        if not self.battery_data:
            raise ValueError("No data loaded!")
        
        return self.battery_data
    
    def extract_features(self):
        """Extract features from loaded data"""
        if self.battery_data is None:
            raise ValueError("No data loaded!")
        
        print("Extracting features...")
        self.features_df = self.feature_extractor.extract_features_from_dataset(self.battery_data)
        return self.features_df
    
    def train_soh_model(self):
        """Train State of Health prediction model"""
        if self.features_df is None:
            self.extract_features()
        
        print("Training SoH prediction model...")
        results = self.soh_predictor.train(self.features_df)
        return results
    
    def train_curve_model(self):
        """Train discharge curve prediction model"""
        if self.battery_data is None:
            raise ValueError("No data loaded!")
        
        print("Training discharge curve prediction model...")
        history = self.curve_predictor.train(self.battery_data)
        return history
    
    def make_random_prediction(self):
        """Make a random prediction for demonstration"""
        if self.battery_data is None or self.soh_predictor.ensemble_model is None:
            raise ValueError("Data not loaded or SoH model not trained!")
        
        # Select random battery and cycle
        battery_name = np.random.choice(list(self.battery_data.keys()))
        battery_cycles = self.battery_data[battery_name]
        cycle_idx = np.random.choice(len(battery_cycles))
        
        selected_cycle = battery_cycles[cycle_idx]
        
        print(f"Selected: {battery_name}, Cycle {cycle_idx + 1}")
        
        # Extract features for this cycle
        features = self.feature_extractor.extract_features_from_cycle(selected_cycle, cycle_idx + 1)
        
        if features is None:
            print("Could not extract features!")
            return None
        
        # Make prediction
        feature_cols = ['time_to_max_temp', 'max_temperature', 'avg_temp_rise_rate', 
                       'time_to_3V', 'initial_voltage_slope', 'cycle_number', 'ambient_temperature']
        
        feature_values = np.array([[features[col] for col in feature_cols]])
        predicted_capacity = self.soh_predictor.predict(feature_values)[0]
        actual_capacity = features['capacity']
        
        # Plot results
        self.plot_cycle_and_prediction(selected_cycle, actual_capacity, predicted_capacity, 
                                     battery_name, cycle_idx + 1)
        
        return {
            'battery_name': battery_name,
            'cycle_number': cycle_idx + 1,
            'actual_capacity': actual_capacity,
            'predicted_capacity': predicted_capacity,
            'error': abs(actual_capacity - predicted_capacity),
            'cycle_data': selected_cycle
        }
    
    def plot_cycle_and_prediction(self, cycle_data, actual_capacity, predicted_capacity, 
                                battery_name, cycle_number):
        """Plot cycle data and capacity prediction"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        time = cycle_data['time'] / 3600  # Convert to hours
        
        # Voltage curve
        ax1.plot(time, cycle_data['voltage'], 'b-', linewidth=2)
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Voltage (V)')
        ax1.set_title(f'{battery_name} - Cycle {cycle_number}: Voltage')
        ax1.grid(True, alpha=0.3)
        
        # Temperature curve
        ax2.plot(time, cycle_data['temperature'], 'r-', linewidth=2)
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_title(f'{battery_name} - Cycle {cycle_number}: Temperature')
        ax2.grid(True, alpha=0.3)
        
        # Current curve
        ax3.plot(time, cycle_data['current'], 'g-', linewidth=2)
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Current (A)')
        ax3.set_title(f'{battery_name} - Cycle {cycle_number}: Current')
        ax3.grid(True, alpha=0.3)
        
        # Capacity prediction
        ax4.bar(['Actual', 'Predicted'], [actual_capacity, predicted_capacity], 
               color=['blue', 'orange'], alpha=0.7)
        ax4.set_ylabel('Capacity (Ah)')
        ax4.set_title('Capacity Prediction')
        ax4.grid(True, alpha=0.3)
        
        # Add error text
        error = abs(actual_capacity - predicted_capacity)
        ax4.text(0.5, max(actual_capacity, predicted_capacity) * 0.9, 
                f'Error: {error:.4f} Ah', ha='center', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'nasa_prediction_{battery_name}_cycle_{cycle_number}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Actual Capacity: {actual_capacity:.4f} Ah")
        print(f"Predicted Capacity: {predicted_capacity:.4f} Ah")
        print(f"Prediction Error: {error:.4f} Ah ({error/actual_capacity*100:.2f}%)")
    
    def save_models(self, soh_path='nasa_soh_model.pkl', curve_path='nasa_curve_model.h5'):
        """Save trained models"""
        if self.soh_predictor.ensemble_model is not None:
            self.soh_predictor.save_model(soh_path)
        
        if self.curve_predictor.model is not None:
            self.curve_predictor.save_model(curve_path)
    
    def load_models(self, soh_path='nasa_soh_model.pkl', curve_path='nasa_curve_model.h5'):
        """Load trained models"""
        try:
            self.soh_predictor.load_model(soh_path)
        except:
            print(f"Could not load SoH model from {soh_path}")
        
        try:
            self.curve_predictor.load_model(curve_path)
        except:
            print(f"Could not load curve model from {curve_path}")

def run_nasa_battery_analysis():
    """Run complete NASA battery analysis pipeline"""
    print("🔋 NASA Battery Analysis - Replicating natskiu/Nasa-Battery")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = NASABatteryAnalyzer()
    
    # Load data
    print("\n1. Loading Data...")
    battery_data = analyzer.load_data(use_sample_data=True)
    
    # Extract features
    print("\n2. Extracting Features...")
    features_df = analyzer.extract_features()
    print(f"Features extracted: {features_df.shape}")
    print(f"Feature columns: {features_df.columns.tolist()}")
    
    # Train SoH model
    print("\n3. Training State of Health Prediction Model...")
    soh_results = analyzer.train_soh_model()
    
    # Train curve prediction model (optional - requires more data)
    try:
        print("\n4. Training Discharge Curve Prediction Model...")
        curve_history = analyzer.train_curve_model()
    except Exception as e:
        print(f"Curve prediction training skipped: {e}")
    
    # Make demonstration prediction
    print("\n5. Making Random Prediction...")
    prediction_result = analyzer.make_random_prediction()
    
    # Save models
    print("\n6. Saving Models...")
    analyzer.save_models()
    
    print("\n✅ NASA Battery Analysis Complete!")
    print(f"Final Test RMSE: {soh_results['test_rmse']:.6f} Ah")
    print(f"Final Test R²: {soh_results['test_r2']:.6f}")
    
    return analyzer

if __name__ == "__main__":
    # Run the complete analysis
    analyzer = run_nasa_battery_analysis()