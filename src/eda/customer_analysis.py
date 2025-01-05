import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class CustomerAnalysis:
    def __init__(self, df):
        """
        Initialize the CustomerAnalysis class with a DataFrame.
        Args:
            df (pd.DataFrame): The input DataFrame.
        """
        self.df = df

    def visualize_customer_metrics(self):
        """Visualize distributions of key features and the sales-per-customer ratio in a 2-column subplot."""
        print("Visualizing feature distributions and sales-per-customer ratio...")

        self.df["SalesPerCustomer"] = self.df["Sales"] / self.df["Customers"]

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))

        # PDistribution of Customers
        sns.histplot(
            self.df["Customers"], kde=True, bins=30, color="skyblue", ax=axes[0]
        )
        axes[0].set_title("Distribution of Customers", fontsize=14)
        axes[0].set_xlabel("Customers")
        axes[0].set_ylabel("Frequency")
        axes[0].grid(True)

        #  Distribution of Sales per Customer
        sns.histplot(
            self.df["SalesPerCustomer"], bins=30, kde=True, color="purple", ax=axes[1]
        )
        axes[1].set_title("Distribution of Sales per Customer", fontsize=14)
        axes[1].set_xlabel("Sales per Customer")
        axes[1].set_ylabel("Frequency")
        axes[1].grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.show()

    def daily_customer_patterns(self):
        """Analyze customer patterns by DayOfWeek."""
        print("Analyzing customer patterns by DayOfWeek...")
        avg_customers = self.df.groupby("DayOfWeek")["Customers"].mean()
        plt.figure(figsize=(8, 5))
        sns.barplot(x=avg_customers.index, y=avg_customers.values, palette="viridis")
        plt.title("Average Customers by Day of the Week")
        plt.xlabel("Day of the Week")
        plt.ylabel("Average Customers")
        plt.grid(True)
        plt.show()

    def promo_customer_patterns(self):
        """Analyze customer behavior during promo periods."""
        print("Comparing customer behavior during promo periods...")
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=self.df, x="Promo", y="Customers", palette="coolwarm")
        plt.title("Customer Distribution: Promo vs. Non-Promo")
        plt.xlabel("Promo")
        plt.ylabel("Number of Customers")
        plt.grid(True)
        plt.show()

    def compare_promo_and_holiday_impact(self):
        """Compare customer counts during promotional periods and holidays."""
        print("Comparing the impact of promotions and holidays on customer traffic...")

        promo_customers = self.df.groupby("Promo")["Customers"].mean()
        holiday_customers = self.df.groupby(self.df["StateHoliday"] != "0")[
            "Customers"
        ].mean()

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))

        # Promo vs No Promo
        sns.barplot(
            x=["No Promo", "Promo"],
            y=promo_customers.values,
            palette="Set2",
            ax=axes[0],
        )
        axes[0].set_title("Average Customers: Promo vs No Promo", fontsize=14)
        axes[0].set_ylabel("Average Customers")
        axes[0].grid(axis="y", linestyle="--", alpha=0.7)

        #  Holidays vs Non-Holidays
        sns.barplot(
            x=["Non-Holiday", "Holiday"],
            y=holiday_customers.values,
            palette="coolwarm",
            ax=axes[1],
        )
        axes[1].set_title("Average Customers: Holiday vs Non-Holiday", fontsize=14)
        axes[1].set_ylabel("Average Customers")
        axes[1].grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.show()

    def top_and_bottom_customer_stores(self, n=10):
        """Identify stores with the highest and lowest average customer numbers."""
        print(f"Analyzing top {n} and bottom 3 stores by average customer count...")

        top_stores = self.df.groupby("Store")["Customers"].mean().nlargest(n)
        bottom_stores = self.df.groupby("Store")["Customers"].mean().nsmallest(n)

        fig, axes = plt.subplots(1, 2, figsize=(8, 5))
        #  Top N Stores
        sns.barplot(
            x=top_stores.index.astype(str),
            y=top_stores.values,
            palette="crest",
            ax=axes[0],
        )
        axes[0].set_title(f"Top {n} Stores by Average Customers", fontsize=14)
        axes[0].set_xlabel("Store ID")
        axes[0].set_ylabel("Average Customers")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].grid(axis="y", linestyle="--", alpha=0.7)

        # Bottom 3 Stores
        sns.barplot(
            x=bottom_stores.index.astype(str),
            y=bottom_stores.values,
            palette="flare",
            ax=axes[1],
        )
        axes[1].set_title(f"Bottom {n} Stores by Average Customers", fontsize=14)
        axes[1].set_xlabel("Store ID")
        axes[1].set_ylabel("Average Customers")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.show()
