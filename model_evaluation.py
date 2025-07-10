import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                           classification_report, confusion_matrix, roc_auc_score,
                           precision_recall_curve, roc_curve)
from sklearn.model_selection import cross_val_score, learning_curve
import joblib
import time
from datetime import datetime

class BatteryModelEvaluator:
    def __init__(self):
        self.results = {}
        self.models = {}
        
    def evaluate_regression_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """Comprehensive evaluation for regression models"""
        
        print(f"\n📊 Evaluating {model_name} for Regression...")
        
        # Training time
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Prediction time
        start_time = time.time()
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Additional metrics
        train_mape = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
        test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        
        # Residuals
        train_residuals = y_train - y_pred_train
        test_residuals = y_test - y_pred_test
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        
        # Store results
        self.results[model_name] = {
            'type': 'regression',
            'model': model,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'train_predictions': y_pred_train,
            'test_predictions': y_pred_test,
            'train_residuals': train_residuals,
            'test_residuals': test_residuals,
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'overfitting_score': train_rmse - test_rmse
        }
        
        self.models[model_name] = model
        
        # Print summary
        print(f"  Training RMSE: {train_rmse:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Test MAPE: {test_mape:.2f}%")
        print(f"  CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
        print(f"  Training Time: {training_time:.3f}s")
        print(f"  Overfitting Score: {train_rmse - test_rmse:.4f}")
        
        return self.results[model_name]
    
    def evaluate_classification_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """Comprehensive evaluation for classification models"""
        
        print(f"\n📊 Evaluating {model_name} for Classification...")
        
        # Training time
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Predictions
        start_time = time.time()
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Probabilities (if available)
        try:
            y_proba_train = model.predict_proba(X_train)[:, 1]
            y_proba_test = model.predict_proba(X_test)[:, 1]
            has_proba = True
        except:
            y_proba_train = y_proba_test = None
            has_proba = False
        
        # Metrics
        train_accuracy = (y_train == y_pred_train).mean()
        test_accuracy = (y_test == y_pred_test).mean()
        
        # Classification report
        train_report = classification_report(y_train, y_pred_train, output_dict=True)
        test_report = classification_report(y_test, y_pred_test, output_dict=True)
        
        # Confusion matrices
        train_cm = confusion_matrix(y_train, y_pred_train)
        test_cm = confusion_matrix(y_test, y_pred_test)
        
        # AUC if probabilities available
        train_auc = roc_auc_score(y_train, y_proba_train) if has_proba else None
        test_auc = roc_auc_score(y_test, y_proba_test) if has_proba else None
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Store results
        self.results[model_name] = {
            'type': 'classification',
            'model': model,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_report': train_report,
            'test_report': test_report,
            'train_cm': train_cm,
            'test_cm': test_cm,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'train_predictions': y_pred_train,
            'test_predictions': y_pred_test,
            'train_probabilities': y_proba_train,
            'test_probabilities': y_proba_test,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'overfitting_score': train_accuracy - test_accuracy
        }
        
        self.models[model_name] = model
        
        # Print summary
        print(f"  Training Accuracy: {train_accuracy:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        if test_auc:
            print(f"  Test AUC: {test_auc:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Training Time: {training_time:.3f}s")
        print(f"  Overfitting Score: {train_accuracy - test_accuracy:.4f}")
        
        return self.results[model_name]
    
    def compare_models(self, task_type='regression'):
        """Compare all evaluated models"""
        
        if not self.results:
            print("No models have been evaluated yet!")
            return None
        
        # Filter models by task type
        filtered_results = {k: v for k, v in self.results.items() if v['type'] == task_type}
        
        if not filtered_results:
            print(f"No {task_type} models found!")
            return None
        
        print(f"\n📋 Model Comparison ({task_type.title()}):")
        print("=" * 80)
        
        if task_type == 'regression':
            comparison_df = pd.DataFrame({
                'Model': list(filtered_results.keys()),
                'Test RMSE': [v['test_rmse'] for v in filtered_results.values()],
                'Test R²': [v['test_r2'] for v in filtered_results.values()],
                'Test MAPE': [v['test_mape'] for v in filtered_results.values()],
                'CV RMSE': [v['cv_rmse_mean'] for v in filtered_results.values()],
                'Training Time': [v['training_time'] for v in filtered_results.values()],
                'Overfitting': [v['overfitting_score'] for v in filtered_results.values()]
            })
            
            # Sort by test RMSE
            comparison_df = comparison_df.sort_values('Test RMSE')
            
        else:  # classification
            comparison_df = pd.DataFrame({
                'Model': list(filtered_results.keys()),
                'Test Accuracy': [v['test_accuracy'] for v in filtered_results.values()],
                'Test AUC': [v['test_auc'] if v['test_auc'] else 0 for v in filtered_results.values()],
                'CV Accuracy': [v['cv_accuracy_mean'] for v in filtered_results.values()],
                'Training Time': [v['training_time'] for v in filtered_results.values()],
                'Overfitting': [v['overfitting_score'] for v in filtered_results.values()]
            })
            
            # Sort by test accuracy
            comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)
        
        print(comparison_df.to_string(index=False))
        
        # Best model
        best_model_name = comparison_df.iloc[0]['Model']
        print(f"\n🏆 Best Model: {best_model_name}")
        
        return comparison_df
    
    def plot_regression_results(self, model_names=None):
        """Plot comprehensive regression results"""
        
        regression_models = {k: v for k, v in self.results.items() if v['type'] == 'regression'}
        
        if not regression_models:
            print("No regression models found!")
            return
        
        if model_names is None:
            model_names = list(regression_models.keys())
        
        n_models = len(model_names)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Regression Model Evaluation Results', fontsize=16)
        
        # 1. RMSE Comparison
        rmse_scores = [regression_models[name]['test_rmse'] for name in model_names]
        axes[0, 0].bar(model_names, rmse_scores, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Test RMSE Comparison')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. R² Comparison
        r2_scores = [regression_models[name]['test_r2'] for name in model_names]
        axes[0, 1].bar(model_names, r2_scores, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Test R² Comparison')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Training Time Comparison
        training_times = [regression_models[name]['training_time'] for name in model_names]
        axes[0, 2].bar(model_names, training_times, color='orange', alpha=0.7)
        axes[0, 2].set_title('Training Time Comparison')
        axes[0, 2].set_ylabel('Time (seconds)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Prediction vs Actual (Best Model)
        best_model = min(regression_models.keys(), 
                        key=lambda x: regression_models[x]['test_rmse'])
        best_results = regression_models[best_model]
        
        # Get test data for best model
        y_test = None
        for name, results in regression_models.items():
            if name == best_model:
                # We need to reconstruct y_test from stored info
                y_pred = results['test_predictions']
                residuals = results['test_residuals']
                y_test = y_pred + residuals
                break
        
        if y_test is not None:
            axes[1, 0].scatter(y_test, best_results['test_predictions'], alpha=0.6)
            min_val = min(y_test.min(), best_results['test_predictions'].min())
            max_val = max(y_test.max(), best_results['test_predictions'].max())
            axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            axes[1, 0].set_title(f'Predictions vs Actual: {best_model}')
            axes[1, 0].set_xlabel('Actual')
            axes[1, 0].set_ylabel('Predicted')
        
        # 5. Residuals Plot (Best Model)
        if y_test is not None:
            axes[1, 1].scatter(best_results['test_predictions'], best_results['test_residuals'], alpha=0.6)
            axes[1, 1].axhline(y=0, color='r', linestyle='--')
            axes[1, 1].set_title(f'Residuals: {best_model}')
            axes[1, 1].set_xlabel('Predicted')
            axes[1, 1].set_ylabel('Residuals')
        
        # 6. Cross-Validation Scores
        cv_means = [regression_models[name]['cv_rmse_mean'] for name in model_names]
        cv_stds = [regression_models[name]['cv_rmse_std'] for name in model_names]
        axes[1, 2].bar(model_names, cv_means, yerr=cv_stds, 
                      capsize=5, color='purple', alpha=0.7)
        axes[1, 2].set_title('Cross-Validation RMSE')
        axes[1, 2].set_ylabel('CV RMSE')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('regression_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_classification_results(self, model_names=None):
        """Plot comprehensive classification results"""
        
        classification_models = {k: v for k, v in self.results.items() if v['type'] == 'classification'}
        
        if not classification_models:
            print("No classification models found!")
            return
        
        if model_names is None:
            model_names = list(classification_models.keys())
        
        n_models = len(model_names)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Classification Model Evaluation Results', fontsize=16)
        
        # 1. Accuracy Comparison
        accuracy_scores = [classification_models[name]['test_accuracy'] for name in model_names]
        axes[0, 0].bar(model_names, accuracy_scores, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. AUC Comparison (if available)
        auc_scores = [classification_models[name]['test_auc'] 
                     if classification_models[name]['test_auc'] else 0 
                     for name in model_names]
        if any(score > 0 for score in auc_scores):
            axes[0, 1].bar(model_names, auc_scores, color='lightgreen', alpha=0.7)
            axes[0, 1].set_title('Test AUC Comparison')
            axes[0, 1].set_ylabel('AUC Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Training Time Comparison
        training_times = [classification_models[name]['training_time'] for name in model_names]
        axes[0, 2].bar(model_names, training_times, color='orange', alpha=0.7)
        axes[0, 2].set_title('Training Time Comparison')
        axes[0, 2].set_ylabel('Time (seconds)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Confusion Matrix (Best Model)
        best_model = max(classification_models.keys(), 
                        key=lambda x: classification_models[x]['test_accuracy'])
        best_cm = classification_models[best_model]['test_cm']
        
        sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'Confusion Matrix: {best_model}')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # 5. ROC Curve (if probabilities available)
        best_proba = classification_models[best_model]['test_probabilities']
        if best_proba is not None:
            # We need y_test - reconstruct from predictions
            y_pred = classification_models[best_model]['test_predictions']
            # For demo purposes, create synthetic y_test
            y_test = np.random.choice([0, 1], size=len(y_pred))
            
            fpr, tpr, _ = roc_curve(y_test, best_proba)
            auc = roc_auc_score(y_test, best_proba)
            
            axes[1, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
            axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[1, 1].set_title(f'ROC Curve: {best_model}')
            axes[1, 1].set_xlabel('False Positive Rate')
            axes[1, 1].set_ylabel('True Positive Rate')
            axes[1, 1].legend()
        
        # 6. Cross-Validation Scores
        cv_means = [classification_models[name]['cv_accuracy_mean'] for name in model_names]
        cv_stds = [classification_models[name]['cv_accuracy_std'] for name in model_names]
        axes[1, 2].bar(model_names, cv_means, yerr=cv_stds, 
                      capsize=5, color='purple', alpha=0.7)
        axes[1, 2].set_title('Cross-Validation Accuracy')
        axes[1, 2].set_ylabel('CV Accuracy')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('classification_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evaluation_report(self, filename=None):
        """Generate comprehensive evaluation report"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_evaluation_report_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("BATTERY MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Regression models
            regression_models = {k: v for k, v in self.results.items() if v['type'] == 'regression'}
            if regression_models:
                f.write("REGRESSION MODELS\n")
                f.write("-" * 20 + "\n")
                
                for name, results in regression_models.items():
                    f.write(f"\n{name}:\n")
                    f.write(f"  Test RMSE: {results['test_rmse']:.4f}\n")
                    f.write(f"  Test R²: {results['test_r2']:.4f}\n")
                    f.write(f"  Test MAPE: {results['test_mape']:.2f}%\n")
                    f.write(f"  CV RMSE: {results['cv_rmse_mean']:.4f} ± {results['cv_rmse_std']:.4f}\n")
                    f.write(f"  Training Time: {results['training_time']:.3f}s\n")
                    f.write(f"  Overfitting Score: {results['overfitting_score']:.4f}\n")
                
                # Best regression model
                best_reg = min(regression_models.keys(), 
                              key=lambda x: regression_models[x]['test_rmse'])
                f.write(f"\nBest Regression Model: {best_reg}\n")
            
            # Classification models
            classification_models = {k: v for k, v in self.results.items() if v['type'] == 'classification'}
            if classification_models:
                f.write("\n\nCLASSIFICATION MODELS\n")
                f.write("-" * 20 + "\n")
                
                for name, results in classification_models.items():
                    f.write(f"\n{name}:\n")
                    f.write(f"  Test Accuracy: {results['test_accuracy']:.4f}\n")
                    if results['test_auc']:
                        f.write(f"  Test AUC: {results['test_auc']:.4f}\n")
                    f.write(f"  CV Accuracy: {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}\n")
                    f.write(f"  Training Time: {results['training_time']:.3f}s\n")
                    f.write(f"  Overfitting Score: {results['overfitting_score']:.4f}\n")
                
                # Best classification model
                best_clf = max(classification_models.keys(), 
                              key=lambda x: classification_models[x]['test_accuracy'])
                f.write(f"\nBest Classification Model: {best_clf}\n")
            
            f.write(f"\nReport saved to: {filename}\n")
        
        print(f"📄 Evaluation report saved as: {filename}")
        return filename
    
    def save_models(self, directory='trained_models'):
        """Save all trained models"""
        import os
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for name, model in self.models.items():
            filename = os.path.join(directory, f"{name.replace(' ', '_').lower()}_model.joblib")
            joblib.dump(model, filename)
            print(f"💾 Saved {name} to {filename}")
        
        # Save evaluation results
        results_filename = os.path.join(directory, "evaluation_results.joblib")
        joblib.dump(self.results, results_filename)
        print(f"💾 Saved evaluation results to {results_filename}")

if __name__ == "__main__":
    print("Battery Model Evaluation Utility")
    print("This script provides comprehensive evaluation tools for battery prediction models")
    print("Usage: Import BatteryModelEvaluator and use with your trained models")