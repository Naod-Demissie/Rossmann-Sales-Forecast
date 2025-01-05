import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


class CorrelationAnalysis:
    def __init__(self, df):
        self.df = df
        print("CorrelationAnalysis initialized with data!")

    def analyze_correlations(self):
        """Explore correlations between Sales and other numerical variables."""
        print("Analyzing correlations with Sales...")
        numeric_df = self.df.select_dtypes(include=["number"])
        sales_corr = numeric_df.corr()["Sales"].sort_values(ascending=False)
        print("Feature correlations with Sales:")
        print(sales_corr)

        # Correlation heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        plt.show()

    def plot_scatter_matrix(self):
        """Generate a scatter matrix to visualize feature pair relationships."""
        print("Generating scatter matrix visualization...")
        features = ["Sales", "Customers", "CompetitionDistance"]
        scatter_matrix(self.df[features], figsize=(12, 8), diagonal="kde", alpha=0.6)
        plt.suptitle("Scatter Matrix of Selected Features")
        plt.show()

    def plot_multivariate_feature_analysis(self):
        """Analyze interactions between promotions, store types, holidays, and competition."""
        print("Performing multivariate analysis...")

        # Pairplot as a standalone plot
        sns.pairplot(
            self.df,
            vars=["Sales", "Customers", "CompetitionDistance"],
            hue="Promo",
            diag_kind="kde",
            palette="Set2",
        )
        plt.suptitle("Multivariate Analysis with Promotions", y=1.02)
        plt.show()

    def plot_customer_sales_relationship(self):
        """Visualize the relationship between Sales and Customers."""
        print("Analyzing relationship between Sales and Customers...")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=self.df, x="Customers", y="Sales", alpha=0.6, ax=ax)
        ax.set_title("Sales vs. Customers")
        ax.set_xlabel("Number of Customers")
        ax.set_ylabel("Sales")
        ax.grid(True)
        plt.show()

    def holiday_analysis(self):
        """Evaluate the impact of holidays on sales."""
        # Separate holiday and non-holiday sales
        holiday_sales = self.df[self.df["StateHoliday"] != "0"]
        school_holiday_sales = self.df[self.df["SchoolHoliday"] == 1]
        non_holiday_sales = self.df[self.df["StateHoliday"] == "0"]

        # Compare sales during holidays and non-holidays
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=[
                holiday_sales["Sales"],
                school_holiday_sales["Sales"],
                non_holiday_sales["Sales"],
            ],
            palette="coolwarm",
            showfliers=False,
        )
        plt.xticks([0, 1, 2], ["State Holidays", "School Holidays", "Non-Holidays"])
        plt.title("Sales Comparison: Holidays vs Non-Holidays", fontsize=14)
        plt.ylabel("Sales")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()
