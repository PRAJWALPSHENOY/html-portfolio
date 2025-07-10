#!/usr/bin/env python3
"""
Battery Health Prediction System - Demo Script

This script demonstrates the complete workflow of the battery health prediction system,
including data loading, preprocessing, model training, evaluation, and prediction.

Run this script to see the system in action with sample data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from battery_health_predictor import BatteryHealthPredictor, run_battery_analysis
from data_exploration import BatteryDataExplorer
from model_evaluation import BatteryModelEvaluator

# Additional imports for demo
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb

def generate_sample_battery_data(n_cycles=1000, n_batteries=3):
    """Generate realistic sample battery data for demonstration"""
    print("🔄 Generating sample battery data...")
    
    np.random.seed(42)
    all_data = []
    
    for battery_id in range(n_batteries):
        cycles = np.arange(1, n_cycles + 1)
        
        # Different degradation patterns for different batteries
        base_degradation_rate = 0.0002 + battery_id * 0.00005
        
        # Simulate capacity degradation with noise
        initial_capacity = 2.0 + np.random.normal(0, 0.1)
        capacity_trend = initial_capacity * (1 - base_degradation_rate * cycles)
        capacity_noise = np.random.normal(0, 0.02, len(cycles))
        capacity = capacity_trend + capacity_noise
        capacity = np.maximum(capacity, 0.5)  # Minimum capacity threshold
        
        # Simulate voltage with degradation correlation
        voltage_base = 3.7 - 0.0001 * cycles
        voltage = voltage_base + np.random.normal(0, 0.05, len(cycles))
        
        # Simulate current with usage patterns
        current_pattern = 1.0 + 0.3 * np.sin(cycles / 50) + np.random.normal(0, 0.1, len(cycles))
        current = np.maximum(current_pattern, 0.1)
        
        # Simulate temperature with seasonal variations
        temperature = 25 + 10 * np.sin(cycles / 365 * 2 * np.pi) + np.random.normal(0, 2, len(cycles))
        
        # Simulate internal resistance (increases with degradation)
        resistance_base = 0.05 + base_degradation_rate * 50 * cycles
        resistance = resistance_base + np.random.normal(0, 0.002, len(cycles))
        
        # Create DataFrame for this battery
        battery_data = pd.DataFrame({
            'battery_id': [f'Battery_{battery_id+1}'] * len(cycles),
            'cycle': cycles,
            'capacity': capacity,
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'resistance': resistance,
            'c_rate': np.random.uniform(0.5, 2.0, len(cycles)),
            'time': cycles * 2.5  # Approximate time in hours
        })
        
        all_data.append(battery_data)
    
    # Combine all battery data
    complete_data = pd.concat(all_data, ignore_index=True)
    
    print(f"✅ Generated {len(complete_data)} data points for {n_batteries} batteries")
    print(f"   Cycles per battery: {n_cycles}")
    print(f"   Features: {complete_data.columns.tolist()}")
    
    return complete_data

def demo_data_exploration(data):
    """Demonstrate data exploration capabilities"""
    print("\n" + "="*60)
    print("🔍 STEP 1: DATA EXPLORATION")
    print("="*60)
    
    # Initialize explorer
    explorer = BatteryDataExplorer(data)
    
    # Run basic analysis
    explorer.basic_info()
    explorer.statistical_summary()
    explorer.missing_value_analysis()
    explorer.outlier_detection()
    explorer.correlation_analysis()
    explorer.battery_specific_analysis()
    
    # Generate visualizations
    print("\n📊 Generating data exploration visualizations...")
    explorer.generate_visualizations(save_plots=True)
    
    # Generate report
    explorer.generate_report('demo_data_exploration_report.txt')
    
    return explorer

def demo_model_training(data):
    """Demonstrate model training and comparison"""
    print("\n" + "="*60)
    print("🤖 STEP 2: MODEL TRAINING & EVALUATION")
    print("="*60)
    
    # Initialize predictor
    predictor = BatteryHealthPredictor()
    
    # Use a subset of data for faster demo
    demo_data = data.sample(n=min(5000, len(data)), random_state=42).copy()
    
    print(f"Using {len(demo_data)} samples for training demo...")
    
    # Preprocess data
    processed_data = predictor.preprocess_data(demo_data)
    
    # Extract features and targets
    features = predictor.extract_features(processed_data)
    targets = predictor.prepare_targets(processed_data, target_type='rul')
    
    if 'rul' not in targets:
        print("⚠️  Could not create RUL targets, using capacity prediction instead")
        targets = predictor.prepare_targets(processed_data, target_type='capacity_prediction')
        target_key = 'capacity'
    else:
        target_key = 'rul'
    
    # Prepare training data
    X = features.fillna(0)
    y = targets[target_key].fillna(0)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Scale features
    X_scaled = predictor.scaler.fit_transform(X)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train models
    print("\n🚀 Training models...")
    
    # Initialize evaluator for comprehensive evaluation
    evaluator = BatteryModelEvaluator()
    
    # Define models to test
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=50, random_state=42, verbosity=0),
        'LightGBM': lgb.LGBMRegressor(n_estimators=50, random_state=42, verbosity=-1)
    }
    
    # Train and evaluate each model
    for name, model in models.items():
        try:
            result = evaluator.evaluate_regression_model(
                model, X_train, X_test, y_train, y_test, name
            )
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
    
    # Compare models
    print("\n📋 Model Comparison:")
    comparison = evaluator.compare_models('regression')
    
    # Generate visualizations
    print("\n📊 Generating model evaluation plots...")
    evaluator.plot_regression_results()
    
    # Generate evaluation report
    evaluator.generate_evaluation_report('demo_model_evaluation_report.txt')
    
    # Save models
    evaluator.save_models('demo_trained_models')
    
    return predictor, evaluator

def demo_predictions(predictor, data):
    """Demonstrate prediction capabilities"""
    print("\n" + "="*60)
    print("🔮 STEP 3: MAKING PREDICTIONS")
    print("="*60)
    
    # Use early cycle data for prediction
    early_cycles = data[data['cycle'] <= 100].copy()
    
    if len(early_cycles) == 0:
        print("⚠️  No early cycle data available for prediction demo")
        return
    
    print(f"Using first 100 cycles ({len(early_cycles)} samples) for prediction...")
    
    try:
        # Make prediction
        prediction, model_name = predictor.predict_battery_life(early_cycles)
        
        print(f"\n🎯 Prediction Results:")
        print(f"   Best Model Used: {model_name}")
        print(f"   Predicted Value: {prediction[0]:.2f}")
        
        # Create prediction visualization
        plt.figure(figsize=(12, 6))
        
        # Plot actual capacity degradation
        for battery_id in data['battery_id'].unique():
            battery_data = data[data['battery_id'] == battery_id]
            plt.plot(battery_data['cycle'], battery_data['capacity'], 
                    label=f'Actual - {battery_id}', alpha=0.7)
        
        plt.axvline(x=100, color='red', linestyle='--', 
                   label='Prediction Point (Cycle 100)')
        plt.xlabel('Cycle Number')
        plt.ylabel('Capacity (Ah)')
        plt.title('Battery Capacity Degradation with Prediction Point')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('demo_prediction_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Prediction visualization saved as 'demo_prediction_visualization.png'")
        
    except Exception as e:
        print(f"❌ Error making predictions: {e}")

def demo_web_app_info():
    """Provide information about the web application"""
    print("\n" + "="*60)
    print("🌐 STEP 4: WEB APPLICATION")
    print("="*60)
    
    print("""
🚀 Launch the interactive web application:

   streamlit run streamlit_app.py

The web app provides:
• 📊 Interactive data upload and analysis
• 🤖 Model training with real-time progress
• 🔮 Easy prediction interface
• 📈 Dynamic visualizations
• 💾 Model saving and loading

Navigate to http://localhost:8501 in your browser once launched.
    """)

def demo_advanced_features():
    """Demonstrate advanced features"""
    print("\n" + "="*60)
    print("⚙️ ADVANCED FEATURES DEMO")
    print("="*60)
    
    print("""
🔧 Advanced Customization Options:

1. Custom Feature Engineering:
   - Add domain-specific battery features
   - Implement time-series features with TSFresh
   - Create ensemble features

2. Model Hyperparameter Tuning:
   - Grid search with cross-validation
   - Bayesian optimization
   - Automated ML with AutoML tools

3. Uncertainty Quantification:
   - Prediction intervals
   - Bayesian neural networks
   - Monte Carlo dropout

4. Real-time Monitoring:
   - Streaming data pipeline
   - Online learning capabilities
   - Alert systems for battery health

5. Production Deployment:
   - Model versioning with MLflow
   - Docker containerization
   - Cloud deployment (AWS, Azure, GCP)
    """)

def main():
    """Main demo function"""
    print("🔋 BATTERY HEALTH PREDICTION SYSTEM - DEMO")
    print("=" * 60)
    print("This demo showcases the complete battery health prediction workflow.")
    print("The system will generate sample data and demonstrate all features.\n")
    
    # Generate sample data
    sample_data = generate_sample_battery_data(n_cycles=500, n_batteries=2)
    
    # Save sample data for later use
    sample_data.to_csv('demo_battery_data.csv', index=False)
    print("💾 Sample data saved as 'demo_battery_data.csv'")
    
    # Step 1: Data Exploration
    explorer = demo_data_exploration(sample_data)
    
    # Step 2: Model Training
    predictor, evaluator = demo_model_training(sample_data)
    
    # Step 3: Predictions
    demo_predictions(predictor, sample_data)
    
    # Step 4: Web App Info
    demo_web_app_info()
    
    # Advanced Features
    demo_advanced_features()
    
    # Summary
    print("\n" + "="*60)
    print("✅ DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("""
📁 Generated Files:
   • demo_battery_data.csv - Sample battery dataset
   • demo_data_exploration_report.txt - Data analysis report
   • demo_model_evaluation_report.txt - Model performance report
   • demo_trained_models/ - Saved ML models
   • Various visualization plots (.png files)

🚀 Next Steps:
   1. Explore the generated files and reports
   2. Launch the web app: streamlit run streamlit_app.py
   3. Try with your own battery data
   4. Customize models and features for your use case

📚 Documentation:
   • README.md - Complete project documentation
   • Code comments - Detailed inline documentation
   • Demo outputs - Generated reports and visualizations
    """)

if __name__ == "__main__":
    main()