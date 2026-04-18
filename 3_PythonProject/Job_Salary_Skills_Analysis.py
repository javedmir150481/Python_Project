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


class DataAnalystSkillsPipeline:
    """
    End-to-end pipeline for:
    1. Loading dataset
    2. Cleaning data
    3. Filtering Data Analyst jobs (US)
    4. Computing skill insights
    5. Visualizing results
    """

    def __init__(self):
        self.df = None
        self.df_da_usa = None
        self.df_da_top_pay = None
        self.df_da_skills = None

    # ------------------- LOAD DATA -------------------
    def load_data(self):
        try:
            dataset = load_dataset('lukebarousse/data_jobs')
            self.df = dataset['train'].to_pandas()

            if self.df.empty:
                raise ValueError("Dataset is empty.")

            logging.info("Data loaded successfully")

        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    # ------------------- CLEAN DATA -------------------
    def clean_data(self):
        try:
            if self.df is None:
                raise ValueError("Data not loaded. Run load_data() first.")

            self.df['job_posted_date'] = pd.to_datetime(
                self.df['job_posted_date'], errors='coerce'
            )

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

        except KeyError as e:
            logging.error(f"Missing column: {e}")
            raise
        except Exception as e:
            logging.error(f"Error cleaning data: {e}")
            raise

    # ------------------- FILTER DATA -------------------
    def filter_data(self):
        try:
            if self.df is None:
                raise ValueError("Data not available.")

            self.df_da_usa = self.df[
                (self.df['job_title_short'] == 'Data Analyst') &
                (self.df['job_country'] == 'United States')
            ].copy()

            self.df_da_usa = self.df_da_usa.dropna(subset=['salary_year_avg'])

            if self.df_da_usa.empty:
                raise ValueError("No Data Analyst US data found.")

            self.df_da_usa = self.df_da_usa.explode('job_skills')

            logging.info("Data filtered successfully")

        except KeyError as e:
            logging.error(f"Column missing: {e}")
            raise
        except Exception as e:
            logging.error(f"Error filtering data: {e}")
            raise

    # ------------------- COMPUTE METRICS -------------------
    def compute_metrics(self):
        try:
            if self.df_da_usa is None:
                raise ValueError("Filtered data not available.")

            self.df_da_top_pay = (
                self.df_da_usa.groupby('job_skills')['salary_year_avg']
                .agg(['count', 'median'])
                .sort_values(by='median', ascending=False)
                .head(10)
            )

            self.df_da_skills = (
                self.df_da_usa.groupby('job_skills')['salary_year_avg']
                .agg(['count', 'median'])
                .sort_values(by='count', ascending=False)
                .head(10)
            )

            self.df_da_skills = self.df_da_skills.sort_values(
                by='median', ascending=False
            )

            logging.info("Metrics computed successfully")

        except Exception as e:
            logging.error(f"Error computing metrics: {e}")
            raise

    # ------------------- VISUALIZATION -------------------
    def visualize(self):
        try:
            if self.df_da_top_pay is None or self.df_da_skills is None:
                raise ValueError("Metrics not computed.")

            fig, ax = plt.subplots(2, 1)

            sns.barplot(
                data=self.df_da_top_pay,
                x='median',
                y=self.df_da_top_pay.index,
                hue='median',
                ax=ax[0],
                palette='dark:b_r'
            )
            ax[0].legend().remove()
            ax[0].set_title('Highest Paid Skills for Data Analysts in the US')
            ax[0].set_xlabel('')
            ax[0].set_ylabel('')
            ax[0].xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'${int(x/1000)}K')
            )

            sns.barplot(
                data=self.df_da_skills,
                x='median',
                y=self.df_da_skills.index,
                hue='median',
                ax=ax[1],
                palette='light:b'
            )
            ax[1].legend().remove()
            ax[1].set_title('Most In-Demand Skills for Data Analysts in the US')
            ax[1].set_xlabel('Median Salary (USD)')
            ax[1].set_ylabel('')
            ax[1].set_xlim(ax[0].get_xlim())
            ax[1].xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'${int(x/1000)}K')
            )

            sns.set_theme(style='ticks')
            plt.tight_layout()
            plt.show()

            logging.info("Visualization completed")

        except Exception as e:
            logging.error(f"Error during visualization: {e}")
            raise


# ------------------- EXECUTION -------------------
if __name__ == "__main__":
    try:
        pipeline = DataAnalystSkillsPipeline()

        pipeline.load_data()
        pipeline.clean_data()
        pipeline.filter_data()
        pipeline.compute_metrics()
        pipeline.visualize()

    except Exception as main_error:
        logging.critical(f"Pipeline failed: {main_error}")