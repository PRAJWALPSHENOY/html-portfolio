#!/usr/bin/env python3
"""
NASA Battery Prediction Script

This script replicates the make_prediction.py functionality from the original project.
It randomly selects a battery and cycle, plots the associated curves, and predicts capacity.

Usage: python make_prediction.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nasa_battery_predictor import NASABatteryAnalyzer
import warnings
warnings.filterwarnings('ignore')

def make_random_prediction():
    """Make a random prediction and display results"""
    print("🔋 NASA Battery Random Prediction")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = NASABatteryAnalyzer()
    
    # Load sample data
    print("Loading sample data...")
    battery_data = analyzer.load_data(use_sample_data=True)
    
    # Extract features
    print("Extracting features...")
    features_df = analyzer.extract_features()
    
    # Train model (quick training for demo)
    print("Training model...")
    soh_results = analyzer.train_soh_model()
    
    # Make random prediction
    print("\nMaking random prediction...")
    result = analyzer.make_random_prediction()
    
    if result:
        print("\n" + "=" * 50)
        print("PREDICTION RESULTS")
        print("=" * 50)
        print(f"Battery: {result['battery_name']}")
        print(f"Cycle: {result['cycle_number']}")
        print(f"Actual Capacity: {result['actual_capacity']:.4f} Ah")
        print(f"Predicted Capacity: {result['predicted_capacity']:.4f} Ah")
        print(f"Absolute Error: {result['error']:.4f} Ah")
        print(f"Relative Error: {result['error']/result['actual_capacity']*100:.2f}%")
        
        # Additional analysis
        cycle_data = result['cycle_data']
        print(f"\nCycle Information:")
        print(f"Discharge Time: {cycle_data['time'][-1]/3600:.2f} hours")
        print(f"Max Temperature: {np.max(cycle_data['temperature']):.2f}°C")
        print(f"Min Voltage: {np.min(cycle_data['voltage']):.2f}V")
        print(f"Average Current: {np.mean(cycle_data['current']):.2f}A")
        
        return result
    else:
        print("Prediction failed!")
        return None

def plot_model_performance(analyzer):
    """Plot overall model performance"""
    if analyzer.features_df is None:
        return
    
    # Plot feature distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    feature_cols = ['time_to_max_temp', 'max_temperature', 'avg_temp_rise_rate', 'time_to_3V']
    
    for i, feature in enumerate(feature_cols):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        for battery in analyzer.features_df['battery_name'].unique():
            battery_data = analyzer.features_df[analyzer.features_df['battery_name'] == battery]
            ax.scatter(battery_data['cycle_number'], battery_data[feature], 
                      label=battery, alpha=0.7, s=20)
        
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel(feature.replace('_', ' ').title())
        ax.set_title(f'{feature.replace("_", " ").title()} vs Cycle')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('nasa_feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_battery_degradation(analyzer):
    """Analyze battery degradation patterns"""
    if analyzer.features_df is None:
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot capacity vs cycle for each battery
    for battery in analyzer.features_df['battery_name'].unique():
        battery_data = analyzer.features_df[analyzer.features_df['battery_name'] == battery]
        battery_data = battery_data.sort_values('cycle_number')
        
        plt.plot(battery_data['cycle_number'], battery_data['capacity'], 
                'o-', label=battery, linewidth=2, markersize=4)
    
    plt.xlabel('Cycle Number')
    plt.ylabel('Capacity (Ah)')
    plt.title('Battery Capacity Degradation Over Cycles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add trend lines
    for battery in analyzer.features_df['battery_name'].unique():
        battery_data = analyzer.features_df[analyzer.features_df['battery_name'] == battery]
        battery_data = battery_data.sort_values('cycle_number')
        
        # Fit polynomial trend
        z = np.polyfit(battery_data['cycle_number'], battery_data['capacity'], 2)
        p = np.poly1d(z)
        plt.plot(battery_data['cycle_number'], p(battery_data['cycle_number']), 
                '--', alpha=0.7, linewidth=1)
    
    plt.tight_layout()
    plt.savefig('nasa_degradation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function"""
    try:
        # Make random prediction
        result = make_random_prediction()
        
        if result:
            # Additional analysis
            analyzer = NASABatteryAnalyzer()
            analyzer.load_data(use_sample_data=True)
            analyzer.extract_features()
            analyzer.train_soh_model()
            
            print("\nGenerating additional analysis plots...")
            plot_model_performance(analyzer)
            analyze_battery_degradation(analyzer)
            
            print("\n✅ Analysis complete! Check the generated plots.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()