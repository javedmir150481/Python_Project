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


class JobSkillsAnalysis:
    """
    A class to load, clean, process, and visualize job skills data.
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.df = None
        self.df_us = None
        self.job_skills_perc = None
        self.top_titles = None

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

            # Convert date column
            self.df['job_posted_date'] = pd.to_datetime(
                self.df['job_posted_date'], errors='coerce'
            )

            # Safe parsing function
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
            logging.error(f"Missing column during cleaning: {e}")
            raise
        except Exception as e:
            logging.error(f"Error during data cleaning: {e}")
            raise

    # =========================
    # Step 3: Filter US Data
    # =========================
    def filter_us_data(self):
        try:
            if self.df is None:
                raise ValueError("Data not available.")

            self.df_us = self.df[
                self.df['job_country'] == 'United States'
            ].copy()

            if self.df_us.empty:
                raise ValueError("No US data found.")

            logging.info("US data filtered successfully")

        except KeyError as e:
            logging.error(f"Missing column: {e}")
            raise
        except Exception as e:
            logging.error(f"Error filtering US data: {e}")
            raise

    # =========================
    # Step 4: Prepare Data
    # =========================
    def prepare_data(self):
        try:
            if self.df_us is None:
                raise ValueError("US data not available. Run filter_us_data() first.")

            df_exploded = self.df_us.explode('job_skills')

            if df_exploded.empty:
                raise ValueError("No data after exploding skills.")

            df_skills_count = (
                df_exploded
                .groupby(['job_skills', 'job_title_short'])
                .size()
                .reset_index(name='skill_count')
                .sort_values(by='skill_count', ascending=False)
            )

            if df_skills_count.empty:
                raise ValueError("Skill aggregation failed.")

            self.top_titles = sorted(
                df_skills_count['job_title_short'].unique().tolist()[:3]
            )

            if not self.top_titles:
                raise ValueError("No job titles found.")

            df_job_count = (
                self.df_us['job_title_short']
                .value_counts()
                .reset_index(name='jobs_count')
            )

            self.job_skills_perc = pd.merge(
                df_skills_count,
                df_job_count,
                on='job_title_short',
                how='left'
            )

            self.job_skills_perc['skill_percent'] = (
                self.job_skills_perc['skill_count'] /
                self.job_skills_perc['jobs_count']
            ) * 100

            logging.info("Data prepared successfully")

        except KeyError as e:
            logging.error(f"Missing column: {e}")
            raise
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            raise

    # =========================
    # Step 5: Visualization
    # =========================
    def plot_skills(self):
        try:
            if self.job_skills_perc is None or self.top_titles is None:
                raise ValueError("Data not ready for visualization.")

            sns.set_theme(style='ticks')

            fig, ax = plt.subplots(len(self.top_titles), 1, figsize=(10, 6))

            if len(self.top_titles) == 1:
                ax = [ax]

            for i, job_title in enumerate(self.top_titles):

                df_plot = (
                    self.job_skills_perc[
                        self.job_skills_perc['job_title_short'] == job_title
                    ]
                    .head(5)
                    .iloc[::-1]
                )

                if df_plot.empty:
                    continue

                sns.barplot(
                    data=df_plot,
                    x='skill_percent',
                    y='job_skills',
                    hue='skill_count',
                    ax=ax[i],
                    palette='dark:b_r'
                )

                ax[i].set_title(job_title)
                ax[i].invert_yaxis()
                ax[i].set_ylabel('')
                ax[i].set_xlabel('')
                ax[i].set_xlim(0, 78)

                legend = ax[i].get_legend()
                if legend:
                    legend.remove()

                if i != len(self.top_titles) - 1:
                    ax[i].set_xticks([])

                for n, v in enumerate(df_plot['skill_percent']):
                    ax[i].text(v + 1, n, f'{v:.0f}%', va='center')

            fig.suptitle(
                'Likelihood of Skills Requested in US Job Postings',
                fontsize=15
            )

            fig.tight_layout(h_pad=0.8)
            plt.show()

            logging.info("Visualization completed")

        except Exception as e:
            logging.error(f"Error during visualization: {e}")
            raise


# =========================
# Usage
# =========================
if __name__ == "__main__":
    try:
        analysis = JobSkillsAnalysis('lukebarousse/data_jobs')

        analysis.load_data()
        analysis.clean_data()
        analysis.filter_us_data()
        analysis.prepare_data()
        analysis.plot_skills()

    except Exception as main_error:
        logging.critical(f"Pipeline failed: {main_error}")