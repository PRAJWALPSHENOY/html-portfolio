import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

class BatteryHealthPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_importance = {}
        
    def load_data(self, file_path, file_type='csv'):
        """Load battery data from various file formats"""
        try:
            if file_type == 'csv':
                data = pd.read_csv(file_path)
            elif file_type == 'excel':
                data = pd.read_excel(file_path)
            elif file_type == 'json':
                data = pd.read_json(file_path)
            else:
                raise ValueError("Unsupported file type. Use 'csv', 'excel', or 'json'")
            
            print(f"Data loaded successfully. Shape: {data.shape}")
            print(f"Columns: {data.columns.tolist()}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, data):
        """Preprocess battery data for ML models"""
        # Handle missing values
        data = data.fillna(data.mean(numeric_only=True))
        
        # Feature engineering for battery health
        processed_data = data.copy()
        
        # Calculate common battery health indicators
        if 'capacity' in data.columns:
            # Capacity fade rate
            processed_data['capacity_fade_rate'] = data['capacity'].diff() / data['capacity'].shift(1)
            
        if 'voltage' in data.columns and 'current' in data.columns:
            # Power calculation
            processed_data['power'] = data['voltage'] * data['current']
            
        if 'temperature' in data.columns:
            # Temperature normalization
            processed_data['temp_normalized'] = (data['temperature'] - data['temperature'].mean()) / data['temperature'].std()
        
        # Remove any remaining NaN values
        processed_data = processed_data.fillna(0)
        
        return processed_data
    
    def extract_features(self, data, cycle_column='cycle', capacity_column='capacity'):
        """Extract features for early cycle prediction"""
        features = []
        
        # Group by cycle and extract statistical features
        if cycle_column in data.columns:
            cycle_stats = data.groupby(cycle_column).agg({
                capacity_column: ['mean', 'std', 'min', 'max'] if capacity_column in data.columns else ['count'],
                'voltage': ['mean', 'std', 'min', 'max'] if 'voltage' in data.columns else ['count'],
                'current': ['mean', 'std', 'min', 'max'] if 'current' in data.columns else ['count'],
                'temperature': ['mean', 'std'] if 'temperature' in data.columns else ['count']
            }).fillna(0)
            
            # Flatten column names
            cycle_stats.columns = ['_'.join(col).strip() for col in cycle_stats.columns]
            features.append(cycle_stats)
        
        # Combine all features
        if features:
            feature_matrix = pd.concat(features, axis=1)
        else:
            # If no cycle column, use the data as is
            feature_matrix = data.select_dtypes(include=[np.number])
        
        return feature_matrix
    
    def prepare_targets(self, data, target_type='rul'):
        """Prepare target variables for different prediction tasks"""
        targets = {}
        
        if target_type == 'rul':
            # Remaining Useful Life (cycles until 80% capacity)
            if 'capacity' in data.columns and 'cycle' in data.columns:
                initial_capacity = data['capacity'].iloc[0]
                threshold = 0.8 * initial_capacity
                
                # Find cycle where capacity drops below threshold
                degraded_cycles = data[data['capacity'] < threshold]['cycle']
                if not degraded_cycles.empty:
                    end_cycle = degraded_cycles.iloc[0]
                    targets['rul'] = end_cycle - data['cycle']
                else:
                    # If threshold not reached, use max cycle
                    targets['rul'] = data['cycle'].max() - data['cycle']
        
        elif target_type == 'capacity_prediction':
            # Predict capacity at future cycles
            if 'capacity' in data.columns:
                targets['capacity'] = data['capacity']
        
        elif target_type == 'classification':
            # Binary classification: healthy vs degraded
            if 'capacity' in data.columns:
                initial_capacity = data['capacity'].iloc[0]
                targets['health_status'] = (data['capacity'] >= 0.8 * initial_capacity).astype(int)
        
        return targets
    
    def train_regression_models(self, X_train, y_train, X_test, y_test):
        """Train and evaluate regression models"""
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred
            }
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
            
            print(f"{name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        self.models.update(results)
        return results
    
    def train_lstm_model(self, X_train, y_train, X_test, y_test, sequence_length=10):
        """Train LSTM model for time series prediction"""
        # Reshape data for LSTM (samples, time steps, features)
        if len(X_train.shape) == 2:
            # If not already a time series, create sequences
            X_train_lstm = self.create_sequences(X_train, sequence_length)
            X_test_lstm = self.create_sequences(X_test, sequence_length)
            y_train_lstm = y_train[sequence_length:]
            y_test_lstm = y_test[sequence_length:]
        else:
            X_train_lstm, X_test_lstm = X_train, X_test
            y_train_lstm, y_test_lstm = y_train, y_test
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Train model
        history = model.fit(X_train_lstm, y_train_lstm, 
                          batch_size=32, epochs=50, 
                          validation_split=0.2, verbose=0)
        
        # Make predictions
        y_pred = model.predict(X_test_lstm)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_lstm, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_lstm, y_pred)
        
        self.models['LSTM'] = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred,
            'history': history
        }
        
        print(f"LSTM - RMSE: {rmse:.4f}, R2: {r2:.4f}")
        return model, history
    
    def create_sequences(self, data, sequence_length):
        """Create sequences for LSTM training"""
        sequences = []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:(i + sequence_length)])
        return np.array(sequences)
    
    def plot_results(self, X_test, y_test):
        """Plot model results and comparisons"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Model comparison
        plt.subplot(2, 3, 1)
        model_names = []
        rmse_scores = []
        
        for name, results in self.models.items():
            if 'rmse' in results:
                model_names.append(name)
                rmse_scores.append(results['rmse'])
        
        plt.bar(model_names, rmse_scores)
        plt.title('Model Comparison (RMSE)')
        plt.xticks(rotation=45)
        plt.ylabel('RMSE')
        
        # Plot 2: Best model predictions vs actual
        best_model_name = min(self.models.keys(), key=lambda x: self.models[x].get('rmse', float('inf')))
        best_predictions = self.models[best_model_name]['predictions']
        
        plt.subplot(2, 3, 2)
        plt.scatter(y_test[:len(best_predictions)], best_predictions, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Best Model: {best_model_name}')
        
        # Plot 3: Feature importance (if available)
        if best_model_name in self.feature_importance:
            plt.subplot(2, 3, 3)
            importance = self.feature_importance[best_model_name]
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
            
            # Plot top 10 features
            top_indices = np.argsort(importance)[-10:]
            plt.barh(range(len(top_indices)), importance[top_indices])
            plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
            plt.title('Top 10 Feature Importance')
        
        # Plot 4: Residuals
        plt.subplot(2, 3, 4)
        residuals = y_test[:len(best_predictions)] - best_predictions.flatten()
        plt.scatter(best_predictions, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # Plot 5: Learning curve (if LSTM)
        if 'LSTM' in self.models and 'history' in self.models['LSTM']:
            plt.subplot(2, 3, 5)
            history = self.models['LSTM']['history']
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('LSTM Learning Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
        
        # Plot 6: R² comparison
        plt.subplot(2, 3, 6)
        r2_scores = [self.models[name].get('r2', 0) for name in model_names]
        plt.bar(model_names, r2_scores)
        plt.title('Model Comparison (R²)')
        plt.xticks(rotation=45)
        plt.ylabel('R² Score')
        
        plt.tight_layout()
        plt.savefig('battery_prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_best_model(self):
        """Return the best performing model"""
        if not self.models:
            return None
        
        best_model_name = min(self.models.keys(), 
                             key=lambda x: self.models[x].get('rmse', float('inf')))
        return best_model_name, self.models[best_model_name]
    
    def predict_battery_life(self, early_cycle_data):
        """Predict battery life using the best model"""
        best_model_name, best_model_info = self.get_best_model()
        
        if best_model_info is None:
            print("No trained models available")
            return None
        
        # Preprocess the input data
        processed_data = self.preprocess_data(early_cycle_data)
        features = self.extract_features(processed_data)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        model = best_model_info['model']
        prediction = model.predict(features_scaled)
        
        return prediction, best_model_name

# Example usage and testing function
def run_battery_analysis(data_file_path):
    """Run complete battery health analysis"""
    
    # Initialize predictor
    predictor = BatteryHealthPredictor()
    
    # Load and preprocess data
    data = predictor.load_data(data_file_path)
    if data is None:
        print("Failed to load data")
        return None
    
    # Preprocess data
    processed_data = predictor.preprocess_data(data)
    
    # Extract features and targets
    features = predictor.extract_features(processed_data)
    targets = predictor.prepare_targets(processed_data, target_type='rul')
    
    if 'rul' not in targets:
        print("Could not prepare target variable. Please check your data format.")
        return None
    
    # Prepare training data
    X = features.fillna(0)
    y = targets['rul'].fillna(0)
    
    # Scale features
    X_scaled = predictor.scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train models
    regression_results = predictor.train_regression_models(X_train, y_train, X_test, y_test)
    
    # Train LSTM if enough data
    if len(X_train) > 50:
        predictor.train_lstm_model(X_train, y_train, X_test, y_test)
    
    # Plot results
    predictor.plot_results(X_test, y_test)
    
    # Get best model
    best_model_name, best_model = predictor.get_best_model()
    print(f"\nBest performing model: {best_model_name}")
    print(f"RMSE: {best_model['rmse']:.4f}")
    print(f"R² Score: {best_model['r2']:.4f}")
    
    return predictor

if __name__ == "__main__":
    # Example with synthetic data if no real data is available
    print("Battery Health Prediction System")
    print("Please provide your data file path to run the analysis")