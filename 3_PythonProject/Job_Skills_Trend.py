# ==============================
# Importing Libraries
# ==============================
import ast
import logging
import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================
# Configure Logging
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ==============================
# Main Class
# ==============================
class SkillTrendAnalysis:

    def __init__(self):
        """Initialize variables"""
        self.df = None
        self.df_da_us = None
        self.df_da_us_pivot = None

    # ==============================
    # Load Data
    # ==============================
    def load_data(self):
        try:
            logging.info("Loading dataset...")
            dataset = load_dataset('lukebarousse/data_jobs')
            self.df = dataset['train'].to_pandas()
            logging.info("Dataset loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise

    # ==============================
    # Data Cleaning
    # ==============================
    def clean_data(self):
        try:
            logging.info("Cleaning data...")

            self.df['job_posted_date'] = pd.to_datetime(self.df['job_posted_date'])

            self.df['job_skills'] = self.df['job_skills'].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) else []
            )

            self.df['job_type_skills'] = self.df['job_type_skills'].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) else {}
            )

            logging.info("Data cleaned successfully.")

        except Exception as e:
            logging.error(f"Error during data cleaning: {e}")
            raise

    # ==============================
    # Filter Data
    # ==============================
    def filter_data(self):
        try:
            logging.info("Filtering Data Analyst jobs in US...")

            self.df_da_us = self.df[
                (self.df['job_title_short'] == 'Data Analyst') &
                (self.df['job_country'] == 'United States')
            ].copy()

            self.df_da_us['job_posted_month_no'] = self.df_da_us['job_posted_date'].dt.month

            logging.info(f"Filtered dataset size: {len(self.df_da_us)}")

        except Exception as e:
            logging.error(f"Error filtering data: {e}")
            raise

    # ==============================
    # Transform Data (Pivot)
    # ==============================
    def transform_data(self):
        try:
            logging.info("Transforming data for trend analysis...")

            df_exploded = self.df_da_us.explode('job_skills')

            df_pivot = df_exploded.pivot_table(
                index='job_posted_month_no',
                columns='job_skills',
                aggfunc='size',
                fill_value=0
            )

            # Add total row for sorting
            df_pivot.loc['Total'] = df_pivot.sum()

            # Sort columns by total descending
            df_pivot = df_pivot[df_pivot.loc['Total'].sort_values(ascending=False).index]

            # Remove total row
            df_pivot = df_pivot.drop('Total')

            self.df_da_us_pivot = df_pivot

            logging.info("Data transformation completed.")

        except Exception as e:
            logging.error(f"Error transforming data: {e}")
            raise

    # ==============================
    # Visualization
    # ==============================
    def plot(self):
        try:
            logging.info("Generating trend visualization...")

            sns.set_theme(style='ticks')
            
            ax = self.df_da_us_pivot.iloc[:, :5].plot(
                kind='line',
                marker='o',
                figsize=(12, 6)
            )

            plt.title('Trending Top Skills for Data Analysts in the US')
            plt.ylabel('Count')
            plt.xlabel('Month')
            plt.xticks(rotation=0)

            plt.tight_layout()
            plt.show()

            logging.info("Visualization completed.")

        except Exception as e:
            logging.error(f"Error during visualization: {e}")
            raise


# ==============================
# Run Pipeline
# ==============================
if __name__ == "__main__":
    try:
        analysis = SkillTrendAnalysis()

        analysis.load_data()
        analysis.clean_data()
        analysis.filter_data()
        analysis.transform_data()
        analysis.plot()

    except Exception as main_error:
        logging.critical(f"Pipeline failed: {main_error}")