import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class PromotionalAnalysis:
    def __init__(self, df):
        """
        Initialize the PromotionalAnalysis class.

        Parameters:
        df (pd.DataFrame): DataFrame containing promotional data.
        """
        self.df = df

    def promo_effectiveness(self):
        """Compare sales and customer numbers during Promo periods versus non-promo periods."""
        print(
            "Analyzing the effectiveness of promotions on sales and customer numbers..."
        )

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))

        # Sales during Promo vs Non-Promo Periods
        sns.boxplot(data=self.df, x="Promo", y="Sales", palette="coolwarm", ax=axes[0])
        axes[0].set_title("Sales During Promo vs Non-Promo Periods")
        axes[0].set_xlabel("Promo Active (1=Yes, 0=No)")
        axes[0].set_ylabel("Sales")

        # Customer numbers during Promo vs Non-Promo Periods
        sns.boxplot(
            data=self.df, x="Promo", y="Customers", palette="viridis", ax=axes[1]
        )
        axes[1].set_title("Customer Numbers During Promo vs Non-Promo Periods")
        axes[1].set_xlabel("Promo Active (1=Yes, 0=No)")
        axes[1].set_ylabel("Customers")

        plt.tight_layout()
        plt.show()

    def promo_effect_on_sales(self):
        """Evaluate the effect of promotions on store sales."""
        print("Evaluating the impact of promotions on store sales...")
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=self.df, x="Promo", y="Sales", palette="cool")
        plt.title("Effect of Promotions on Sales")
        plt.xlabel("Promotion Active (1=Yes, 0=No)")
        plt.ylabel("Sales")
        plt.show()

    def promo2_effectiveness(self):
        """Evaluate the impact of Promo2 and its attributes on sales."""
        print("Evaluating the impact of Promo2 on sales...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        # Sales Distribution by Promo2 Start Week
        sns.boxplot(
            data=self.df, x="Promo2SinceWeek", y="Sales", palette="cool", ax=axes[0]
        )
        axes[0].set_title("Sales Distribution by Promo2 Start Week")
        axes[0].set_xlabel("Promo2 Start Week")
        axes[0].set_ylabel("Sales")
        axes[0].tick_params(axis="x", rotation=45)

        # Sales Distribution by Promo2 Start Year
        sns.boxplot(
            data=self.df, x="Promo2SinceYear", y="Sales", palette="Set3", ax=axes[1]
        )
        axes[1].set_title("Sales Distribution by Promo2 Start Year")
        axes[1].set_xlabel("Promo2 Start Year")
        axes[1].set_ylabel("Sales")
        axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

    def promo_interval_analysis(self):
        """Analyze the impact of PromoInterval on sales."""
        print("Analyzing the impact of different PromoIntervals on sales...")
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=self.df, x="PromoInterval", y="Sales", palette="Spectral")
        plt.title("Sales Distribution by PromoInterval")
        plt.xlabel("Promo Interval")
        plt.ylabel("Sales")
        plt.show()

    def optimal_promo_timing(self):
        """Find the most effective timing for promotions based on Promo2 attributes."""
        print("Determining the optimal timing for promotions...")
        promo_data = self.df[self.df["Promo2"] == 1]
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=promo_data,
            x="Promo2SinceWeek",
            y="Sales",
            hue="PromoInterval",
            marker="o",
            palette="husl",
        )
        plt.title("Sales Trends by Promo2 Start Week and Interval")
        plt.xlabel("Promo2 Start Week")
        plt.ylabel("Sales")
        plt.legend(title="Promo Interval", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xticks(rotation=45)
        plt.show()

    def plot_detailed_feature_interactions(self):
        """Visualize interactions between holidays, promotions, store types, and competition."""
        print(
            "Visualizing feature interactions between holidays, promotions, and store attributes..."
        )

        # Sales by Promotions and Holidays
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sns.boxplot(
            data=self.df,
            x="Promo",
            y="Sales",
            hue="StateHoliday",
            palette="viridis",
            ax=axes[0],
        )
        axes[0].set_title("Sales by Promotions and State Holidays")
        axes[0].set_xlabel("Promotion (1 = Active, 0 = Inactive)")
        axes[0].set_ylabel("Sales")
        axes[0].legend(title="State Holiday")
        axes[0].grid(True)

        # Sales by Store Type and Promotion Status
        sns.barplot(
            data=self.df,
            x="StoreType",
            y="Sales",
            hue="Promo",
            palette="magma",
            ax=axes[1],
        )
        axes[1].set_title("Sales by Store Type and Promotion Status")
        axes[1].set_xlabel("Store Type")
        axes[1].set_ylabel("Average Sales")
        axes[1].legend(title="Promotion Active")
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
