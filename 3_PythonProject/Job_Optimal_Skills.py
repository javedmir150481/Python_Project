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
from adjustText import adjust_text


# =========================
# Configure Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class OptimalSkillsAnalysis:
    """
    Pipeline to identify most optimal (high demand + high salary) skills
    for Data Analysts in the US.
    """

    def __init__(self):
        self.df = None
        self.df_da_us = None
        self.df_skills = None
        self.df_high_demand = None

    # =========================
    # LOAD DATA
    # =========================
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

    # =========================
    # CLEAN DATA
    # =========================
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

        except Exception as e:
            logging.error(f"Error cleaning data: {e}")
            raise

    # =========================
    # FILTER DATA
    # =========================
    def filter_data(self):
        try:
            if self.df is None:
                raise ValueError("Data not available.")

            self.df_da_us = self.df[
                (self.df['job_title_short'] == 'Data Analyst') &
                (self.df['job_country'] == 'United States')
            ].copy()

            self.df_da_us = self.df_da_us.dropna(subset=['salary_year_avg'])

            if self.df_da_us.empty:
                raise ValueError("No Data Analyst US data found.")

            logging.info("Data filtered successfully")

        except Exception as e:
            logging.error(f"Error filtering data: {e}")
            raise

    # =========================
    # PREPARE DATA
    # =========================
    def prepare_data(self):
        try:
            if self.df_da_us is None:
                raise ValueError("Filtered data not available.")

            # Explode skills
            df_exploded = self.df_da_us.explode('job_skills')

            # Aggregate skills
            df_skills = (
                df_exploded.groupby('job_skills')['salary_year_avg']
                .agg(['count', 'median'])
                .sort_values(by='count', ascending=False)
            )

            df_skills = df_skills.rename(
                columns={'count': 'skill_count', 'median': 'median_salary'}
            )

            total_jobs = len(self.df_da_us)

            df_skills['skill_percent'] = (
                df_skills['skill_count'] / total_jobs * 100
            )

            df_skills = df_skills[df_skills['skill_count'] > 0].round(0)

            self.df_skills = df_skills

            # Filter high-demand skills
            skill_limit = 5
            self.df_high_demand = df_skills[
                df_skills['skill_percent'] > skill_limit
            ]

            if self.df_high_demand.empty:
                raise ValueError("No high-demand skills found.")

            logging.info("Data prepared successfully")

        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            raise

    # =========================
    # VISUALIZATION
    # =========================
    def visualize(self):
        try:
            if self.df_high_demand is None:
                raise ValueError("Data not ready for visualization.")

            plt.scatter(
                self.df_high_demand['skill_percent'],
                self.df_high_demand['median_salary']
            )

            plt.xlabel('Percent of Data Analyst Jobs')
            plt.ylabel('Median Salary ($USD)')
            plt.title('Most Optimal Skills for Data Analysts in the US')

            ax = plt.gca()
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, pos: f'${int(y/1000)}K')
            )

            # Add labels
            texts = []
            for i, skill in enumerate(self.df_high_demand.index):
                texts.append(
                    plt.text(
                        self.df_high_demand['skill_percent'].iloc[i],
                        self.df_high_demand['median_salary'].iloc[i],
                        " " + skill
                    )
                )

            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))

            plt.show()

            logging.info("Visualization completed")

        except Exception as e:
            logging.error(f"Error during visualization: {e}")
            raise


# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    try:
        pipeline = OptimalSkillsAnalysis()

        pipeline.load_data()
        pipeline.clean_data()
        pipeline.filter_data()
        pipeline.prepare_data()
        pipeline.visualize()

    except Exception as main_error:
        logging.critical(f"Pipeline failed: {main_error}")