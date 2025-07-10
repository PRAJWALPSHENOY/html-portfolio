import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class BatteryDataExplorer:
    def __init__(self, data):
        """Initialize with battery dataset"""
        self.data = data
        self.numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
    def basic_info(self):
        """Print basic information about the dataset"""
        print("="*60)
        print("BATTERY DATASET OVERVIEW")
        print("="*60)
        print(f"Dataset Shape: {self.data.shape}")
        print(f"Memory Usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Numeric Columns: {len(self.numeric_cols)}")
        print(f"Non-Numeric Columns: {self.data.shape[1] - len(self.numeric_cols)}")
        
        print("\nColumn Information:")
        print("-" * 40)
        for col in self.data.columns:
            dtype = self.data[col].dtype
            null_count = self.data[col].isnull().sum()
            null_pct = (null_count / len(self.data)) * 100
            print(f"{col:20} | {str(dtype):15} | {null_count:6} nulls ({null_pct:5.1f}%)")
    
    def statistical_summary(self):
        """Generate comprehensive statistical summary"""
        print("\n" + "="*60)
        print("STATISTICAL SUMMARY")
        print("="*60)
        
        if self.numeric_cols:
            # Basic statistics
            desc = self.data[self.numeric_cols].describe()
            print("\nBasic Statistics:")
            print(desc)
            
            # Additional statistics
            print("\nAdditional Statistics:")
            additional_stats = pd.DataFrame({
                'Skewness': self.data[self.numeric_cols].skew(),
                'Kurtosis': self.data[self.numeric_cols].kurtosis(),
                'Variance': self.data[self.numeric_cols].var(),
                'Range': self.data[self.numeric_cols].max() - self.data[self.numeric_cols].min()
            })
            print(additional_stats)
    
    def missing_value_analysis(self):
        """Analyze missing values in the dataset"""
        print("\n" + "="*60)
        print("MISSING VALUE ANALYSIS")
        print("="*60)
        
        missing_data = self.data.isnull().sum()
        missing_pct = (missing_data / len(self.data)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing Percentage': missing_pct.values
        })
        
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if missing_df.empty:
            print("✅ No missing values found in the dataset!")
        else:
            print("Missing values summary:")
            print(missing_df)
        
        return missing_df
    
    def outlier_detection(self):
        """Detect outliers using IQR method"""
        print("\n" + "="*60)
        print("OUTLIER DETECTION (IQR Method)")
        print("="*60)
        
        outlier_summary = []
        
        for col in self.numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(self.data)) * 100
            
            outlier_summary.append({
                'Column': col,
                'Outlier Count': outlier_count,
                'Outlier Percentage': outlier_pct,
                'Lower Bound': lower_bound,
                'Upper Bound': upper_bound
            })
        
        outlier_df = pd.DataFrame(outlier_summary)
        print(outlier_df)
        
        return outlier_df
    
    def correlation_analysis(self):
        """Analyze correlations between features"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        if len(self.numeric_cols) > 1:
            corr_matrix = self.data[self.numeric_cols].corr()
            
            # Find high correlations (above 0.7)
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr_pairs.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })
            
            if high_corr_pairs:
                print("High correlations (|r| > 0.7):")
                for pair in high_corr_pairs:
                    print(f"{pair['Feature 1']} - {pair['Feature 2']}: {pair['Correlation']:.3f}")
            else:
                print("No high correlations found (|r| > 0.7)")
            
            return corr_matrix
        else:
            print("Not enough numeric columns for correlation analysis")
            return None
    
    def battery_specific_analysis(self):
        """Perform battery-specific data analysis"""
        print("\n" + "="*60)
        print("BATTERY-SPECIFIC ANALYSIS")
        print("="*60)
        
        # Check for common battery columns
        battery_cols = {
            'capacity': ['capacity', 'cap', 'discharge_capacity'],
            'voltage': ['voltage', 'volt', 'terminal_voltage'],
            'current': ['current', 'curr', 'discharge_current'],
            'temperature': ['temperature', 'temp', 'ambient_temp'],
            'cycle': ['cycle', 'cycle_number', 'cycle_index'],
            'resistance': ['resistance', 'internal_resistance', 'ir']
        }
        
        found_cols = {}
        for category, possible_names in battery_cols.items():
            for col in self.data.columns:
                if any(name.lower() in col.lower() for name in possible_names):
                    found_cols[category] = col
                    break
        
        print("Detected battery-related columns:")
        for category, col_name in found_cols.items():
            print(f"  {category.capitalize()}: {col_name}")
        
        # Capacity analysis
        if 'capacity' in found_cols:
            cap_col = found_cols['capacity']
            print(f"\nCapacity Analysis ({cap_col}):")
            print(f"  Initial Capacity: {self.data[cap_col].iloc[0]:.3f}")
            print(f"  Final Capacity: {self.data[cap_col].iloc[-1]:.3f}")
            print(f"  Capacity Fade: {((self.data[cap_col].iloc[0] - self.data[cap_col].iloc[-1]) / self.data[cap_col].iloc[0] * 100):.2f}%")
            print(f"  Min Capacity: {self.data[cap_col].min():.3f}")
            print(f"  Max Capacity: {self.data[cap_col].max():.3f}")
        
        # Cycle analysis
        if 'cycle' in found_cols:
            cycle_col = found_cols['cycle']
            print(f"\nCycle Analysis ({cycle_col}):")
            print(f"  Total Cycles: {self.data[cycle_col].max()}")
            print(f"  Cycle Range: {self.data[cycle_col].min()} - {self.data[cycle_col].max()}")
        
        # Temperature analysis
        if 'temperature' in found_cols:
            temp_col = found_cols['temperature']
            print(f"\nTemperature Analysis ({temp_col}):")
            print(f"  Mean Temperature: {self.data[temp_col].mean():.2f}°C")
            print(f"  Temperature Range: {self.data[temp_col].min():.2f} - {self.data[temp_col].max():.2f}°C")
            print(f"  Temperature Std: {self.data[temp_col].std():.2f}°C")
        
        return found_cols
    
    def generate_visualizations(self, save_plots=True):
        """Generate comprehensive visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig_count = 1
        
        # 1. Distribution plots for numeric variables
        if len(self.numeric_cols) > 0:
            n_cols = min(3, len(self.numeric_cols))
            n_rows = (len(self.numeric_cols) + n_cols - 1) // n_cols
            
            plt.figure(figsize=(15, 5 * n_rows))
            for i, col in enumerate(self.numeric_cols):
                plt.subplot(n_rows, n_cols, i + 1)
                self.data[col].hist(bins=30, alpha=0.7, edgecolor='black')
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
            
            plt.tight_layout()
            if save_plots:
                plt.savefig(f'battery_distributions_{fig_count}.png', dpi=300, bbox_inches='tight')
                fig_count += 1
            plt.show()
        
        # 2. Correlation heatmap
        if len(self.numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            corr_matrix = self.data[self.numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.1)
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            if save_plots:
                plt.savefig(f'battery_correlation_{fig_count}.png', dpi=300, bbox_inches='tight')
                fig_count += 1
            plt.show()
        
        # 3. Battery-specific plots
        battery_cols = self.battery_specific_analysis()
        
        # Capacity degradation plot
        if 'capacity' in battery_cols and 'cycle' in battery_cols:
            plt.figure(figsize=(12, 6))
            plt.plot(self.data[battery_cols['cycle']], self.data[battery_cols['capacity']], 
                    'b-', linewidth=2, alpha=0.8)
            plt.title('Battery Capacity Degradation Over Cycles')
            plt.xlabel('Cycle Number')
            plt.ylabel('Capacity')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_plots:
                plt.savefig(f'capacity_degradation_{fig_count}.png', dpi=300, bbox_inches='tight')
                fig_count += 1
            plt.show()
        
        # Voltage vs Current scatter plot
        if 'voltage' in battery_cols and 'current' in battery_cols:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.data[battery_cols['voltage']], self.data[battery_cols['current']], 
                       alpha=0.6, s=10)
            plt.title('Voltage vs Current Relationship')
            plt.xlabel('Voltage')
            plt.ylabel('Current')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_plots:
                plt.savefig(f'voltage_current_scatter_{fig_count}.png', dpi=300, bbox_inches='tight')
                fig_count += 1
            plt.show()
        
        # Temperature distribution over time
        if 'temperature' in battery_cols and 'cycle' in battery_cols:
            plt.figure(figsize=(12, 6))
            plt.plot(self.data[battery_cols['cycle']], self.data[battery_cols['temperature']], 
                    'r-', alpha=0.7, linewidth=1)
            plt.title('Temperature Variation Over Cycles')
            plt.xlabel('Cycle Number')
            plt.ylabel('Temperature (°C)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_plots:
                plt.savefig(f'temperature_cycles_{fig_count}.png', dpi=300, bbox_inches='tight')
                fig_count += 1
            plt.show()
        
        # 4. Box plots for outlier visualization
        if len(self.numeric_cols) > 0:
            n_cols = min(3, len(self.numeric_cols))
            n_rows = (len(self.numeric_cols) + n_cols - 1) // n_cols
            
            plt.figure(figsize=(15, 5 * n_rows))
            for i, col in enumerate(self.numeric_cols):
                plt.subplot(n_rows, n_cols, i + 1)
                self.data.boxplot(column=col, ax=plt.gca())
                plt.title(f'Box Plot: {col}')
            
            plt.tight_layout()
            if save_plots:
                plt.savefig(f'battery_boxplots_{fig_count}.png', dpi=300, bbox_inches='tight')
                fig_count += 1
            plt.show()
        
        print(f"✅ Generated {fig_count-1} visualization plots")
    
    def generate_report(self, filename='battery_data_report.txt'):
        """Generate a comprehensive text report"""
        print(f"\n📊 Generating comprehensive report: {filename}")
        
        with open(filename, 'w') as f:
            f.write("BATTERY DATASET EXPLORATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic info
            f.write("1. DATASET OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Shape: {self.data.shape}\n")
            f.write(f"Memory Usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
            f.write(f"Numeric Columns: {len(self.numeric_cols)}\n\n")
            
            # Column info
            f.write("2. COLUMN INFORMATION\n")
            f.write("-" * 20 + "\n")
            for col in self.data.columns:
                dtype = self.data[col].dtype
                null_count = self.data[col].isnull().sum()
                null_pct = (null_count / len(self.data)) * 100
                f.write(f"{col}: {dtype}, {null_count} nulls ({null_pct:.1f}%)\n")
            f.write("\n")
            
            # Statistical summary
            if self.numeric_cols:
                f.write("3. STATISTICAL SUMMARY\n")
                f.write("-" * 20 + "\n")
                f.write(str(self.data[self.numeric_cols].describe()))
                f.write("\n\n")
            
            # Missing values
            missing_df = self.missing_value_analysis()
            f.write("4. MISSING VALUES\n")
            f.write("-" * 20 + "\n")
            if missing_df.empty:
                f.write("No missing values found.\n")
            else:
                f.write(str(missing_df))
            f.write("\n\n")
            
            # Outliers
            outlier_df = self.outlier_detection()
            f.write("5. OUTLIERS (IQR Method)\n")
            f.write("-" * 20 + "\n")
            f.write(str(outlier_df))
            f.write("\n\n")
            
            # Battery-specific analysis
            f.write("6. BATTERY-SPECIFIC INSIGHTS\n")
            f.write("-" * 20 + "\n")
            battery_cols = self.battery_specific_analysis()
            for category, col_name in battery_cols.items():
                f.write(f"{category.capitalize()}: {col_name}\n")
            f.write("\n")
            
            f.write("Report generated successfully!\n")
        
        print(f"✅ Report saved as {filename}")

def explore_battery_data(data_path, file_type='csv'):
    """Main function to run complete battery data exploration"""
    
    # Load data
    try:
        if file_type == 'csv':
            data = pd.read_csv(data_path)
        elif file_type == 'excel':
            data = pd.read_excel(data_path)
        elif file_type == 'json':
            data = pd.read_json(data_path)
        else:
            raise ValueError("Unsupported file type")
        
        print(f"✅ Data loaded successfully from {data_path}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Initialize explorer
    explorer = BatteryDataExplorer(data)
    
    # Run comprehensive analysis
    explorer.basic_info()
    explorer.statistical_summary()
    explorer.missing_value_analysis()
    explorer.outlier_detection()
    explorer.correlation_analysis()
    explorer.battery_specific_analysis()
    explorer.generate_visualizations()
    explorer.generate_report()
    
    return explorer

if __name__ == "__main__":
    print("Battery Data Explorer")
    print("This script provides comprehensive exploration of battery datasets")
    print("Usage: python data_exploration.py")
    print("Make sure to update the data_path variable with your dataset path")
    
    # Example usage
    # explorer = explore_battery_data('your_battery_data.csv', 'csv')