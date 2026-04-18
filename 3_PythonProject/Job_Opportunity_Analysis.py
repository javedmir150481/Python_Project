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
class JobConditionAnalysis:

    def __init__(self):
        """Initialize variables"""
        self.df = None
        self.df_da_usa = None
        self.dict_column = {
            'job_work_from_home': 'Work from Home Offered',
            'job_no_degree_mention': 'Degree Requirement',
            'job_health_insurance': 'Health Insurance Offered'
        }

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

            self.df_da_usa = self.df[
                (self.df['job_title_short'] == 'Data Analyst') &
                (self.df['job_country'] == 'United States')
            ].copy()

            logging.info(f"Filtered dataset size: {len(self.df_da_usa)}")

        except Exception as e:
            logging.error(f"Error filtering data: {e}")
            raise

    # ==============================
    # Visualization
    # ==============================
    def plot(self):
        try:
            logging.info("Generating pie charts...")

            sns.set_theme(style='ticks')

            fig, ax = plt.subplots(1, 3, figsize=(12, 5))

            for i, (column, title) in enumerate(self.dict_column.items()):

                # Handle missing values safely
                value_counts = self.df_da_usa[column].fillna(False).value_counts()

                # Ensure consistent order (True, False)
                value_counts = value_counts.reindex([True, False], fill_value=0)

                ax[i].pie(
                    value_counts,
                    labels=['True', 'False'],
                    autopct='%1.1f%%',
                    startangle=90
                )

                ax[i].set_title(title)

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
        analysis = JobConditionAnalysis()

        analysis.load_data()
        analysis.clean_data()
        analysis.filter_data()
        analysis.plot()

    except Exception as main_error:
        logging.critical(f"Pipeline failed: {main_error}")