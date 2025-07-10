#!/usr/bin/env python3
"""
Simple test script to verify the battery health prediction project structure.
This script checks if all components are properly set up and can be imported.
"""

import sys
import os

def test_project_structure():
    """Test if all required files exist"""
    print("🔍 Testing Project Structure...")
    
    required_files = [
        'battery_health_predictor.py',
        'streamlit_app.py',
        'data_exploration.py',
        'model_evaluation.py',
        'requirements.txt',
        'README.md',
        'demo_notebook.py'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        return False
    else:
        print("\n✅ All required files are present!")
        return True

def test_code_syntax():
    """Test if Python files have valid syntax"""
    print("\n🔍 Testing Code Syntax...")
    
    python_files = [
        'battery_health_predictor.py',
        'streamlit_app.py',
        'data_exploration.py',
        'model_evaluation.py',
        'demo_notebook.py'
    ]
    
    syntax_errors = []
    for file in python_files:
        try:
            with open(file, 'r') as f:
                code = f.read()
            compile(code, file, 'exec')
            print(f"✅ {file} - Syntax OK")
        except SyntaxError as e:
            print(f"❌ {file} - Syntax Error: {e}")
            syntax_errors.append(file)
        except Exception as e:
            print(f"⚠️  {file} - Could not test: {e}")
    
    if syntax_errors:
        print(f"\n⚠️  Files with syntax errors: {syntax_errors}")
        return False
    else:
        print("\n✅ All Python files have valid syntax!")
        return True

def test_imports():
    """Test basic imports without external dependencies"""
    print("\n🔍 Testing Basic Imports...")
    
    # Test standard library imports
    try:
        import pandas as pd
        print("✅ pandas import - OK")
    except ImportError:
        print("❌ pandas not available")
    
    try:
        import numpy as np
        print("✅ numpy import - OK")
    except ImportError:
        print("❌ numpy not available")
    
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib import - OK")
    except ImportError:
        print("❌ matplotlib not available")
    
    try:
        import sklearn
        print("✅ scikit-learn import - OK")
    except ImportError:
        print("❌ scikit-learn not available")

def create_sample_data():
    """Create a simple sample dataset for testing"""
    print("\n🔄 Creating Sample Data...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create simple sample data
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'cycle': range(1, n_samples + 1),
            'capacity': 2.0 - 0.0002 * np.arange(n_samples) + np.random.normal(0, 0.05, n_samples),
            'voltage': 3.7 + np.random.normal(0, 0.1, n_samples),
            'current': 1.0 + np.random.normal(0, 0.2, n_samples),
            'temperature': 25 + np.random.normal(0, 5, n_samples),
            'resistance': 0.05 + 0.00001 * np.arange(n_samples) + np.random.normal(0, 0.005, n_samples)
        })
        
        # Save sample data
        data.to_csv('test_battery_data.csv', index=False)
        print(f"✅ Created test dataset with {len(data)} samples")
        print(f"   Saved as 'test_battery_data.csv'")
        
        # Display basic statistics
        print("\n📊 Sample Data Statistics:")
        print(data.describe())
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without full dependencies"""
    print("\n🔍 Testing Basic Functionality...")
    
    try:
        # Test if we can read the sample data
        import pandas as pd
        
        if os.path.exists('test_battery_data.csv'):
            data = pd.read_csv('test_battery_data.csv')
            print(f"✅ Successfully loaded test data: {data.shape}")
            
            # Basic data analysis
            print(f"   Columns: {data.columns.tolist()}")
            print(f"   Capacity range: {data['capacity'].min():.3f} - {data['capacity'].max():.3f}")
            print(f"   Cycle range: {data['cycle'].min()} - {data['cycle'].max()}")
            
            return True
        else:
            print("❌ Test data file not found")
            return False
            
    except Exception as e:
        print(f"❌ Error testing functionality: {e}")
        return False

def display_project_info():
    """Display project information and next steps"""
    print("\n" + "="*60)
    print("🔋 BATTERY HEALTH PREDICTION PROJECT")
    print("="*60)
    
    print("""
📊 Project Components:
   • battery_health_predictor.py - Main ML pipeline
   • streamlit_app.py - Interactive web application
   • data_exploration.py - Data analysis tools
   • model_evaluation.py - Model evaluation utilities
   • demo_notebook.py - Complete demo script
   • requirements.txt - Project dependencies
   • README.md - Comprehensive documentation

🚀 Getting Started:
   1. Install dependencies: pip install -r requirements.txt
   2. Run demo: python demo_notebook.py
   3. Launch web app: streamlit run streamlit_app.py
   4. Explore documentation: README.md

🎯 Key Features:
   • Multiple ML models (Linear, RF, XGBoost, LightGBM, LSTM)
   • Interactive Streamlit web interface
   • Comprehensive data exploration
   • Model comparison and evaluation
   • Real-time battery life predictions

📚 Use Cases:
   • Electric vehicle battery monitoring
   • Research and development
   • Predictive maintenance
   • Battery performance analysis
    """)

def main():
    """Main test function"""
    print("🧪 BATTERY HEALTH PREDICTION PROJECT TEST")
    print("=" * 50)
    
    # Run all tests
    structure_ok = test_project_structure()
    syntax_ok = test_code_syntax()
    test_imports()
    sample_data_ok = create_sample_data()
    functionality_ok = test_basic_functionality()
    
    # Summary
    print("\n" + "="*50)
    print("📋 TEST SUMMARY")
    print("="*50)
    
    tests = {
        "Project Structure": structure_ok,
        "Code Syntax": syntax_ok,
        "Sample Data Creation": sample_data_ok,
        "Basic Functionality": functionality_ok
    }
    
    for test_name, result in tests.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} - {status}")
    
    all_passed = all(tests.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("The battery health prediction project is ready to use!")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
    
    # Display project info
    display_project_info()
    
    return all_passed

if __name__ == "__main__":
    main()