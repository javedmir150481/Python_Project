# =========================
# Importing Libraries
# =========================
import ast
import logging
import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# Configure Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataJobAnalysis:
    """
    This class handles:
    1. Loading dataset
    2. Cleaning data
    3. Filtering and transforming data
    4. Visualizing salary distributions
    """

    def __init__(self):
        """Initialize class variables"""
        self.df = None
        self.df_us = None
        self.df_us_top_titles = None

    # =========================
    # LOAD DATA
    # =========================
    def load_data(self):
        """Loads dataset with error handling"""
        try:
            dataset = load_dataset('lukebarousse/data_jobs')
            self.df = dataset['train'].to_pandas()

            if self.df.empty:
                raise ValueError("Loaded dataset is empty.")

            logging.info("Data loaded successfully")

        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise

    # =========================
    # CLEAN DATA
    # =========================
    def clean_data(self):
        """Cleans and preprocesses data safely"""
        try:
            if self.df is None:
                raise ValueError("Dataframe is not loaded. Call load_data() first.")

            # Convert job posted date
            self.df['job_posted_date'] = pd.to_datetime(
                self.df['job_posted_date'], errors='coerce'
            )

            # Safe parsing function
            def safe_literal_eval(value, default):
                try:
                    return ast.literal_eval(value) if pd.notna(value) else default
                except (ValueError, SyntaxError):
                    return default

            # Apply safe parsing
            self.df['job_skills'] = self.df['job_skills'].apply(
                lambda x: safe_literal_eval(x, [])
            )

            self.df['job_type_skills'] = self.df['job_type_skills'].apply(
                lambda x: safe_literal_eval(x, {})
            )

            logging.info("Data cleaned successfully")

        except KeyError as e:
            logging.error(f"Missing expected column: {e}")
            raise
        except Exception as e:
            logging.error(f"Error during data cleaning: {e}")
            raise

    # =========================
    # FILTER DATA
    # =========================
    def filter_us_data(self):
        """Filters US data with validation"""
        try:
            if self.df is None:
                raise ValueError("Dataframe is not available.")

            self.df_us = self.df[
                (self.df['job_country'] == 'United States')
            ].dropna(subset=['salary_year_avg'])

            if self.df_us.empty:
                raise ValueError("No US data found after filtering.")

            logging.info("US data filtered successfully")

        except KeyError as e:
            logging.error(f"Column not found: {e}")
            raise
        except Exception as e:
            logging.error(f"Error filtering US data: {e}")
            raise

    # =========================
    # PREPARE DATA
    # =========================
    def prepare_top_job_titles(self):
        """Prepares top job titles safely"""
        try:
            if self.df_us is None:
                raise ValueError("US data not available. Run filter_us_data() first.")

            job_titles = self.df_us['job_title_short'].value_counts().index[:6].tolist()

            if not job_titles:
                raise ValueError("No job titles found.")

            self.df_us_top_titles = self.df_us[
                self.df_us['job_title_short'].isin(job_titles)
            ]

            if self.df_us_top_titles.empty:
                raise ValueError("Filtered dataset for top job titles is empty.")

            logging.info("Top job titles prepared")

        except KeyError as e:
            logging.error(f"Column missing: {e}")
            raise
        except Exception as e:
            logging.error(f"Error preparing job titles: {e}")
            raise

    # =========================
    # VISUALIZATION
    # =========================
    def plot_salary_distribution(self):
        """Plots salary distribution with error handling"""
        try:
            if self.df_us_top_titles is None:
                raise ValueError("Data not ready for plotting.")

            if self.df_us_top_titles.empty:
                raise ValueError("No data available to plot.")

            # Compute job order
            job_order = self.df_us_top_titles.groupby('job_title_short')[
                'salary_year_avg'
            ].median().sort_values(ascending=False).index

            sns.set_theme(style='ticks')

            sns.boxplot(
                data=self.df_us_top_titles,
                x='salary_year_avg',
                y='job_title_short',
                order=job_order
            )

            sns.despine()

            plt.title('Salary Distributions of Data Jobs in the US')
            plt.xlabel('Yearly Salary (USD)')
            plt.ylabel('')
            plt.xlim(0, 600000)

            ticks_x = plt.FuncFormatter(lambda x, pos: f'${int(x/1000)}K')
            plt.gca().xaxis.set_major_formatter(ticks_x)

            plt.show()

            logging.info("Plot generated successfully")

        except Exception as e:
            logging.error(f"Error during plotting: {e}")
            raise


# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    try:
        analysis = DataJobAnalysis()

        analysis.load_data()
        analysis.clean_data()
        analysis.filter_us_data()
        analysis.prepare_top_job_titles()
        analysis.plot_salary_distribution()

    except Exception as main_error:
        logging.critical(f"Pipeline failed: {main_error}")