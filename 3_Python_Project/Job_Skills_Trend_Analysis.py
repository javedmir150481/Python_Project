# =========================
# Importing Libraries
# =========================
import ast
import logging
import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# =========================
# Configure Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class SkillTrendAnalyzer:
    """
    Class to analyze and visualize trending skills
    for Data Analysts in the US.
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.df = None
        self.df_da_us = None
        self.df_percent = None

    # =========================
    # Step 1: Load Data
    # =========================
    def load_data(self):
        try:
            dataset = load_dataset(self.dataset_name)
            self.df = dataset['train'].to_pandas()

            if self.df.empty:
                raise ValueError("Loaded dataset is empty.")

            logging.info("Data loaded successfully")

        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise

    # =========================
    # Step 2: Clean Data
    # =========================
    def clean_data(self):
        try:
            if self.df is None:
                raise ValueError("Data not loaded. Run load_data() first.")

            self.df['job_posted_date'] = pd.to_datetime(
                self.df['job_posted_date'], errors='coerce'
            )

            # Safe parsing
            def safe_literal_eval(value, default):
                try:
                    return ast.literal_eval(value) if pd.notna(value) else default
                except (ValueError, SyntaxError):
                    return default

            self.df['job_skills'] = self.df['job_skills'].apply(
                lambda x: safe_literal_eval(x, [])
            )

            self.df['job_type_skills'] = self.df['job_type_skills'].apply(
                lambda x: safe_literal_eval(x, {})
            )

            logging.info("Data cleaned successfully")

        except Exception as e:
            logging.error(f"Error during cleaning: {e}")
            raise

    # =========================
    # Step 3: Filter Data
    # =========================
    def filter_data(self):
        try:
            if self.df is None:
                raise ValueError("Data not available.")

            self.df_da_us = self.df[
                (self.df['job_title_short'] == 'Data Analyst') &
                (self.df['job_country'] == 'United States')
            ].copy()

            if self.df_da_us.empty:
                raise ValueError("No Data Analyst US data found.")

            self.df_da_us['job_posted_month_no'] = (
                self.df_da_us['job_posted_date'].dt.month
            )

            logging.info("Data filtered successfully")

        except Exception as e:
            logging.error(f"Error filtering data: {e}")
            raise

    # =========================
    # Step 4: Prepare Data
    # =========================
    def prepare_data(self):
        try:
            if self.df_da_us is None:
                raise ValueError("Filtered data not available.")

            df_exploded = self.df_da_us.explode('job_skills')

            df_pivot = df_exploded.pivot_table(
                index='job_posted_month_no',
                columns='job_skills',
                aggfunc='size',
                fill_value=0
            )

            sorted_cols = df_pivot.sum().sort_values(ascending=False).index
            df_pivot = df_pivot[sorted_cols]

            df_totals = self.df_da_us.groupby('job_posted_month_no').size()

            df_percent = df_pivot.iloc[:12].div(df_totals / 100, axis=0)

            df_percent = df_percent.reset_index()
            df_percent['job_posted_month'] = df_percent['job_posted_month_no'].apply(
                lambda x: pd.to_datetime(str(x), format='%m').strftime('%b')
            )

            df_percent = df_percent.set_index('job_posted_month')
            df_percent = df_percent.drop(columns='job_posted_month_no')

            self.df_percent = df_percent

            logging.info("Data prepared successfully")

        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            raise

    # =========================
    # Step 5: Plot Data
    # =========================
    def plot_trends(self, top_n=5):
        try:
            if self.df_percent is None or self.df_percent.empty:
                raise ValueError("Data not ready for plotting.")

            sns.set_theme(style='ticks')

            df_plot = self.df_percent.iloc[:, :top_n]

            sns.lineplot(data=df_plot, dashes=False, legend=False, palette='tab10')
            sns.despine()

            plt.title('Trending Top Skills for Data Analysts in the US')
            plt.ylabel('Likelihood in Job Posting')
            plt.xlabel('2023')

            plt.gca().yaxis.set_major_formatter(PercentFormatter(decimals=0))

            last_values = df_plot.iloc[-1].sort_values()

            for i, (col, val) in enumerate(last_values.items()):
                plt.text(len(df_plot) - 1 + 0.2, val + i * 1.2, col)

            plt.show()

            logging.info("Visualization completed successfully")

        except Exception as e:
            logging.error(f"Error during plotting: {e}")
            raise


# =========================
# Usage
# =========================
if __name__ == "__main__":
    try:
        analysis = SkillTrendAnalyzer('lukebarousse/data_jobs')

        analysis.load_data()
        analysis.clean_data()
        analysis.filter_data()
        analysis.prepare_data()
        analysis.plot_trends(top_n=5)

    except Exception as main_error:
        logging.critical(f"Pipeline failed: {main_error}")