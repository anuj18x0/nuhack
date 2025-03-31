import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import streamlit as st

class SoilAnalysisSystem:
    def __init__(self):
        self.data = None
        self.nutrient_ranges = {
            'N': {'low': 0, 'optimal': 40, 'high': 80},
            'P': {'low': 0, 'optimal': 20, 'high': 40},
            'K': {'low': 150, 'optimal': 250, 'high': 350},
            'pH': {'low': 5.5, 'optimal': 6.5, 'high': 7.5},
            'organic_matter': {'low': 2, 'optimal': 5, 'high': 10}
        }
        self.possible_nutrients = [
            'N', 'P', 'K', 'pH', 'organic_matter', 
            'nitrogen', 'phosphorus', 'potassium', 
            'calcium', 'magnesium', 'sulfur', 'zinc', 'iron',
            'n', 'p', 'k', 'ph', 'organic', 'om',
            'nitro', 'phos', 'pot', 'ca', 'mg', 's', 'zn', 'fe',
            'moisture', 'temp', 'temperature', 'conductivity', 'ec'
        ]
        self.column_mapping = {}
    
    def load_data(self, file_path, file_type='csv'):
        """Optimized function to load soil data from CSV or Excel."""
        loaders = {'csv': pd.read_csv, 'excel': pd.read_excel, 'xlsx': pd.read_excel}
        try:
            self.data = loaders.get(file_type.lower(), lambda x: None)(file_path)
            if self.data is None:
                raise ValueError("Unsupported file type. Use 'csv' or 'excel.")

            return self.data.head()
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def set_column_mapping(self, mapping):
        """
        Set a mapping from dataset column names to standard nutrient names
        Example: {'soil_n': 'N', 'soil_p': 'P', 'soil_k': 'K', 'acidity': 'pH'}
        """
        self.column_mapping = mapping
        st.write(f"Column mapping set: {mapping}")
    
    def get_mapped_columns(self, columns=None):
        """Get the mapped column names or try to infer them"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        if columns is not None:
            return columns
        
        if self.column_mapping:
            return [col for col in self.data.columns if col in self.column_mapping]
        
        inferred_cols = [col for col in self.data.columns 
                         if col.lower() in [n.lower() for n in self.possible_nutrients]]
        
        if not inferred_cols:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            st.warning("Could not identify specific nutrient columns. Using all numeric columns instead.")
            return numeric_cols
        
        return inferred_cols
    
    def get_standardized_name(self, column):
        """Get the standardized nutrient name for a column"""
        if column in self.column_mapping:
            return self.column_mapping[column]
        return column
    
    def preprocess_data(self):
        """Clean and preprocess the data."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        missing_values = self.data.isnull().sum()
        st.write(f"Missing values in each column:\n{missing_values}")
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].apply(lambda col: col.fillna(col.median()))
        
        initial_rows = self.data.shape[0]
        self.data = self.data.drop_duplicates()
        st.write(f"Removed {initial_rows - self.data.shape[0]} duplicate rows")
        
        self.data = self.data.applymap(lambda x: str(x) if isinstance(x, (list, dict)) else x)
        
        return self.data.describe()
    
    def analyze_nutrients(self, nutrient_cols=None):
        """Analyze nutrient levels in soil samples and add results to the dataset."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        nutrient_cols = self.get_mapped_columns(nutrient_cols)
        
        if not nutrient_cols:
            st.warning("No nutrient columns identified. Analysis may not be meaningful.")
            nutrient_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()

        analysis = {}
        for col in nutrient_cols:
            if col in self.data.columns:
                mean_val = self.data[col].mean()
                analysis[col] = {
                    "Mean": mean_val,
                    "Median": self.data[col].median(),
                    "Min": self.data[col].min(),
                    "Max": self.data[col].max(),
                    "Status": self._determine_nutrient_status(col, mean_val)
                }
                self.data[f"{col}_status"] = self._determine_nutrient_status(col, mean_val)
                self.data[f"{col}_mean"] = mean_val

        return pd.DataFrame(analysis)
    
    def _determine_nutrient_status(self, col, mean_val):
        """Determine the status of a nutrient based on predefined ranges."""
        std_name = self.get_standardized_name(col)
        key = std_name if std_name in self.nutrient_ranges else std_name.upper()

        if key in self.nutrient_ranges:
            ranges = self.nutrient_ranges[key]
            if mean_val < ranges['low']:
                return "Very Low"
            elif mean_val < ranges['optimal']:
                return "Low"
            elif mean_val <= ranges['high']:
                return "Optimal"
            else:
                return "High"
        return "Unknown"
    
    def _calculate_nutrient_score(self, col, mean_val):
        """Calculate a nutrient score based on predefined ranges."""
        std_name = self.get_standardized_name(col)
        key = std_name if std_name in self.nutrient_ranges else std_name.upper()

        if key in self.nutrient_ranges:
            ranges = self.nutrient_ranges[key]
            if mean_val < ranges['low']:
                return 25
            elif mean_val < ranges['optimal']:
                return 50
            elif mean_val <= ranges['high']:
                return 75
            else:
                return 100
        return 50
    
    def calculate_soil_health_score(self, nutrient_cols=None):
        """Calculate an overall soil health score more efficiently."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        nutrient_cols = self.get_mapped_columns(nutrient_cols)
        
        scores = [
            self._calculate_nutrient_score(col, self.data[col].mean())
            for col in nutrient_cols if col in self.data.columns
        ]

        return round(np.mean(scores), 2) if scores else 50.0
    
    def generate_recommendations(self, nutrient_analysis=None):
        """Generate recommendations based on soil analysis using a mapping approach."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        if nutrient_analysis is None:
            nutrient_analysis = self.analyze_nutrients()

        status_dict = nutrient_analysis.loc['Status'].to_dict()
        
        recommendations_map = {
            'N': "Increase nitrogen with fertilizers or compost.",
            'P': "Increase phosphorus with phosphate fertilizers or bone meal.",
            'K': "Increase potassium with potash fertilizers or wood ash.",
            'pH_low': "Soil is acidic. Add lime to raise pH.",
            'pH_high': "Soil is alkaline. Add sulfur to lower pH.",
            'organic_matter': "Increase organic matter with compost or manure."
        }

        recommendations = [
            recommendations_map.get(nutrient, f"Monitor {nutrient}, current status: {status}")
            for nutrient, status in status_dict.items()
            if (nutrient in recommendations_map) or (nutrient == 'pH' and status in ['Low', 'High'])
        ]

        if len(recommendations) < 2:
            recommendations.extend([
                "Implement crop rotation to improve soil structure.",
                "Conduct soil testing annually.",
                "Maintain proper soil moisture levels."
            ])

        return recommendations
    
    def visualize_all_columns(self):
        """Generate graphs for all columns in the dataset."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        st.write("Visualizing all columns in the dataset...")
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        fig, axes = plt.subplots(nrows=(len(numeric_cols) + 1) // 2, ncols=2, figsize=(14, 3 * ((len(numeric_cols) + 1) // 2)))
        axes = np.ravel(axes)

        for ax, col in zip(axes, numeric_cols):
            sns.histplot(self.data[col], kde=True, ax=ax)
            ax.set_title(f'Distribution of {col}')
        
        for ax in axes[len(numeric_cols):]:
            ax.axis('off')

        st.pyplot(fig)
    
    def visualize_correlations(self, columns=None):
        """Visualize correlations between soil properties"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        if columns is not None:
            data_to_plot = self.data[columns].select_dtypes(include=[np.number])
        else:
            data_to_plot = self.data.select_dtypes(include=[np.number])
        
        if data_to_plot.shape[1] < 2:
            st.warning("Not enough numeric columns for correlation analysis.")
            return
        
        corr_matrix = data_to_plot.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
        plt.title('Correlation Between Soil Properties')
        plt.tight_layout()
        st.pyplot(plt)
    
    def run_pca_analysis(self, columns=None):
        """Run PCA analysis to identify patterns in soil data"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        if columns is not None:
            data_to_analyze = self.data[columns].select_dtypes(include=[np.number])
        else:
            data_to_analyze = self.data.select_dtypes(include=[np.number])
        
        if data_to_analyze.shape[1] < 2:
            st.warning("Not enough numeric columns for PCA analysis.")
            return None
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_to_analyze)
        
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        pca_df = pd.DataFrame(
            data=pca_result[:, 0:min(2, pca_result.shape[1])],
            columns=['PC1', 'PC2'] if pca_result.shape[1] >= 2 else ['PC1']
        )
        
        if pca_result.shape[1] >= 2:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x='PC1', y='PC2', data=pca_df)
            plt.title('PCA of Soil Properties')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(plt)
        
        return pd.DataFrame({
            'Principal Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
            'Explained Variance Ratio': pca.explained_variance_ratio_,
            'Cumulative Variance Ratio': np.cumsum(pca.explained_variance_ratio_)
        })
    
    def set_nutrient_ranges(self, ranges):
        """
        Set custom nutrient ranges for analysis
        Example: {'N': {'low': 20, 'optimal': 50, 'high': 80}}
        """
        self.nutrient_ranges.update(ranges)
        st.write(f"Updated nutrient ranges: {ranges}")
    
    def generate_report(self):
        """Generate a comprehensive soil analysis report."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        try:
            nutrient_analysis = self.analyze_nutrients()
            health_score = self.calculate_soil_health_score()
            recommendations = self.generate_recommendations(nutrient_analysis)
            
            st.write("SOIL ANALYSIS REPORT")
            st.write(f"Number of samples analyzed: {self.data.shape[0]}")
            st.write(f"Overall soil health score: {health_score}/100")
            
            st.write("NUTRIENT ANALYSIS:")
            st.dataframe(nutrient_analysis)
            
            st.write("RECOMMENDATIONS:")
            for rec in recommendations:
                st.write(f"- {rec}")
            
            return {
                'health_score': health_score,
                'nutrient_analysis': nutrient_analysis,
                'recommendations': recommendations
            }
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            return None
    
    def visualize_scatter_comparison(self):
        """Generate scatter plots comparing original data with analyzed data."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        st.write("Scatter Plot Comparison: Original vs Analyzed Data")
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns

        # Filter columns that have corresponding "_mean" columns
        analyzed_cols = [col for col in numeric_cols if f"{col}_mean" in self.data.columns]

        if not analyzed_cols:
            st.warning("No analyzed columns found for comparison.")
            return

        fig, axes = plt.subplots(nrows=len(analyzed_cols), ncols=1, figsize=(10, 5 * len(analyzed_cols)))

        if len(analyzed_cols) == 1:
            axes = [axes]  # Ensure axes is iterable for a single plot

        for ax, col in zip(axes, analyzed_cols):
            ax.scatter(self.data[col], self.data[f"{col}_mean"], alpha=0.6, label=f"{col} vs {col}_mean")
            ax.set_xlabel(f"Original {col}")
            ax.set_ylabel(f"Analyzed {col}_mean")
            ax.set_title(f"Scatter Plot: {col} vs {col}_mean")
            ax.legend()

        plt.tight_layout()
        st.pyplot(fig)

def main():
    st.title("Soil Analysis System")
    st.sidebar.title("Navigation")
    options = ["Home", "Load and Full Analysis"]
    choice = st.sidebar.radio("Go to", options)

    soil_analyzer = SoilAnalysisSystem()

    if choice == "Home":
        st.write("Welcome to the Soil Analysis System. Use the sidebar to navigate through the app.")

    elif choice == "Load and Full Analysis":
        st.subheader("Load Soil Data and Perform Full Analysis")
        file = st.file_uploader("Upload your soil data file (CSV or Excel)", type=["csv", "xlsx"])
        file_type = st.selectbox("Select file type", ["csv", "excel"])
        if file:
            soil_analyzer.load_data(file, file_type)
            st.write("Data Preview:")
            st.dataframe(soil_analyzer.data.head())

            st.write("Preprocessing data...")
            summary = soil_analyzer.preprocess_data()
            st.write("Data Summary:")
            st.dataframe(summary)

            st.write("Analyzing nutrient levels...")
            nutrient_analysis = soil_analyzer.analyze_nutrients()
            st.write("Nutrient Analysis:")
            st.dataframe(nutrient_analysis)

            st.write("Visualizing all columns in the dataset...")
            soil_analyzer.visualize_all_columns()

            st.write("Generating scatter plot comparison...")
            soil_analyzer.visualize_scatter_comparison()

            st.write("Generating report...")
            report = soil_analyzer.generate_report()
            if report:
                st.write("Soil Health Score:", report['health_score'])
                st.write("Nutrient Analysis:")
                st.dataframe(report['nutrient_analysis'])
                st.write("Recommendations:")
                for rec in report['recommendations']:
                    st.write(f"- {rec}")

if __name__ == "__main__":
    main()
