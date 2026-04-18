# ==============================
# Importing Libraries
# ==============================
import ast
import logging
import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from adjustText import adjust_text


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
class DataJobAnalysis:

    def __init__(self):
        """Initialize variables"""
        self.df = None
        self.df_da_us = None
        self.df_da_us_skills = None
        self.df_da_us_tech_high_demand = None

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

            # Convert string to list safely
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
            logging.info("Filtering US Data Analyst jobs...")

            self.df_da_us = self.df[
                (self.df['job_title_short'] == 'Data Analyst') &
                (self.df['job_country'] == 'United States')
            ].copy()

            self.df_da_us = self.df_da_us.dropna(subset='salary_year_avg')

            logging.info(f"Filtered dataset size: {len(self.df_da_us)}")

        except Exception as e:
            logging.error(f"Error filtering data: {e}")
            raise

    # ==============================
    # Skill Processing
    # ==============================
    def process_skills(self):
        try:
            logging.info("Processing skills...")

            df_exploded = self.df_da_us.explode('job_skills')

            df_skills = (
                df_exploded.groupby('job_skills')['salary_year_avg']
                .agg(['count', 'median'])
                .sort_values(by='count', ascending=False)
            )

            df_skills = df_skills.rename(
                columns={'count': 'skill_count', 'median': 'median_salary'}
            )

            total_jobs = len(self.df_da_us)

            df_skills['skill_percent'] = (df_skills['skill_count'] / total_jobs) * 100

            self.df_da_us_skills = df_skills[df_skills['skill_count'] > 0].round(0)

            logging.info("Skill processing completed.")

        except Exception as e:
            logging.error(f"Error processing skills: {e}")
            raise

    # ==============================
    # Technology Mapping
    # ==============================
    def map_technology(self):
        try:
            logging.info("Mapping skills to technology...")

            df_technology = self.df[['job_type_skills', 'job_skills']].copy()

            df_technology = df_technology.explode('job_type_skills')
            df_technology = df_technology.explode('job_skills')

            df_technology = df_technology.dropna(subset=['job_skills', 'job_type_skills'])

            # ✅ FIX: get most frequent category per skill
            df_technology = (
                df_technology.groupby(['job_skills', 'job_type_skills'])
                .size()
                .reset_index(name='count')
            )
            # Keep dominant category
            df_technology = df_technology.sort_values('count', ascending=False)
            df_technology = df_technology.drop_duplicates(subset='job_skills')

            df_merged = self.df_da_us_skills.merge(
                df_technology[['job_skills', 'job_type_skills']],
                on='job_skills',
                how='left'
            )

            # Filter high demand skills
            self.df_da_us_tech_high_demand = df_merged[
                df_merged['skill_percent'] > 5
            ]

            logging.info("Technology mapping completed.")

        except Exception as e:
            logging.error(f"Error mapping technology: {e}")
            raise

    # ==============================
    # Visualization
    # ==============================
    def plot(self):
        try:
            logging.info("Generating visualization...")

            sns.set_theme(style='ticks')
            plt.figure(figsize=(10, 6))

            sns.scatterplot(
                data=self.df_da_us_tech_high_demand,
                x='skill_percent',
                y='median_salary',
                hue='job_type_skills'
            )

            sns.despine()

            # Add labels with offset
            texts = []
            for x, y, txt in zip(
                self.df_da_us_tech_high_demand['skill_percent'],
                self.df_da_us_tech_high_demand['median_salary'],
                self.df_da_us_tech_high_demand['job_skills']
            ):
                texts.append(
                    plt.text(
                        x,
                        y,
                        txt,
                        fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                    )
                )

            # Adjust text
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))
            

            # Labels & formatting
            plt.xlabel('Percent of Data Analyst Jobs')
            plt.ylabel('Median Yearly Salary')
            plt.title('Most Optimal Skills for Data Analysts in the US')
            plt.legend(title='Technology')

            ax = plt.gca()
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f'${int(y/1000)}K')
            )
            ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))

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
        analysis = DataJobAnalysis()
        
        analysis.load_data()
        analysis.clean_data()
        analysis.filter_data()
        analysis.process_skills()
        analysis.map_technology()
        analysis.plot()

    except Exception as main_error:
        logging.critical(f"Pipeline failed: {main_error}")