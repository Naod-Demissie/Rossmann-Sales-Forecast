import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class SalesAnalysis:
    def __init__(self, df):
        self.df = df
        print("SalesVariabilityAnalysis initialized with data!")

    def sales_distribution(self):
        """Visualize the distribution of Sales to identify variability and outliers."""
        print("Visualizing sales distribution...")
        plt.figure(figsize=(8, 5))
        sns.histplot(self.df["Sales"], kde=True, bins=50, color="blue")
        plt.title("Sales Distribution")
        plt.xlabel("Sales")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

    def sales_by_store_type(self):
        """Segment sales data by different store types."""
        print("Analyzing sales distribution across store types...")
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=self.df, x="StoreType", y="Sales", palette="coolwarm")
        plt.title("Sales Distribution by Store Type")
        plt.xlabel("Store Type")
        plt.ylabel("Sales")
        plt.grid(True)
        plt.show()

    def sales_by_promotion(self):
        """Segment sales data by promotion periods."""
        print("Analyzing sales distribution during promotion periods...")
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=self.df, x="Promo", y="Sales", palette="viridis")
        plt.title("Sales Distribution: Promo vs Non-Promo")
        plt.xlabel("Promotion")
        plt.ylabel("Sales")
        plt.grid(True)
        plt.show()

    def sales_by_holidays(self):
        """Segment sales data by holidays (StateHoliday and SchoolHoliday) using subplots."""
        print("Analyzing sales distribution during holidays...")

        # Create a figure with two subplots (1 row, 2 columns)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

        # State holidays sales boxplot
        sns.boxplot(
            data=self.df, x="StateHoliday", y="Sales", palette="magma", ax=axes[0]
        )
        axes[0].set_title("Sales Distribution by State Holidays")
        axes[0].set_xlabel("State Holiday")
        axes[0].set_ylabel("Sales")
        axes[0].grid(True)

        # School holidays sales boxplot
        sns.boxplot(
            data=self.df, x="SchoolHoliday", y="Sales", palette="coolwarm", ax=axes[1]
        )
        axes[1].set_title("Sales Distribution by School Holidays")
        axes[1].set_xlabel("School Holiday")
        axes[1].grid(True)

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

    def holiday_sales_impact(self):
        """Analyze sales during holidays versus non-holidays."""
        print("Analyzing the impact of holidays on sales...")
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=self.df, x="StateHoliday", y="Sales", palette="Spectral")
        plt.title("Sales During Holidays vs Non-Holidays")
        plt.xlabel("State Holiday")
        plt.ylabel("Sales")
        plt.show()

    def holiday_vs_non_holiday_sales(self):
        """Compare sales on holidays versus non-holidays using subplots."""
        print("Comparing sales on holidays versus non-holidays...")

        fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

        # State holidays sales boxplot
        sns.boxplot(
            data=self.df, x="StateHoliday", y="Sales", palette="coolwarm", ax=axes[0]
        )
        axes[0].set_title("Sales During State Holidays vs Non-Holidays")
        axes[0].set_xlabel("State Holiday (0 = Non-Holiday, a/b/c = Holiday Types)")
        axes[0].set_ylabel("Sales")

        # School holidays sales boxplot
        sns.boxplot(
            data=self.df, x="SchoolHoliday", y="Sales", palette="viridis", ax=axes[1]
        )
        axes[1].set_title("Sales During School Holidays vs Non-Holidays")
        axes[1].set_xlabel("School Holiday (1 = Yes, 0 = No)")

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()

    def high_low_sales_analysis(self):
        """Identify stores or days with unusually high or low sales."""
        print("Identifying high and low sales outliers...")
        high_sales_threshold = self.df["Sales"].quantile(0.95)
        low_sales_threshold = self.df["Sales"].quantile(0.05)

        high_sales = self.df[self.df["Sales"] > high_sales_threshold]
        low_sales = self.df[self.df["Sales"] < low_sales_threshold]

        print(f"High Sales Threshold: {high_sales_threshold}")
        print(f"Low Sales Threshold: {low_sales_threshold}")

        print(f"Number of High Sales Days: {len(high_sales)}")
        print(f"Number of Low Sales Days: {len(low_sales)}")

        # Visualize high and low sales
        plt.figure(figsize=(11, 5))
        sns.boxplot(data=self.df, y="Sales", x="StoreType", palette="coolwarm")
        plt.title("High and Low Sales Distribution by Store Type")
        plt.axhline(
            y=high_sales_threshold,
            color="red",
            linestyle="--",
            label="High Sales Threshold",
        )
        plt.axhline(
            y=low_sales_threshold,
            color="green",
            linestyle="--",
            label="Low Sales Threshold",
        )
        plt.legend()
        plt.xlabel("Store Type")
        plt.ylabel("Sales")
        plt.grid(True)
        plt.show()

    def sales_variability_by_day(self):
        """Analyze daily sales variability."""
        print("Analyzing daily sales variability...")
        avg_sales_by_day = self.df.groupby("DayOfWeek")["Sales"].mean()
        plt.figure(figsize=(8, 5))
        sns.barplot(
            x=avg_sales_by_day.index, y=avg_sales_by_day.values, palette="viridis"
        )

        plt.title("Average Sales by Day of the Week with Variability")
        plt.xlabel("Day of the Week")
        plt.ylabel("Sales")
        plt.grid(True)
        plt.show()

    def competition_distance_effect(self):
        """Analyze the effect of CompetitionDistance on Sales."""
        print("Analyzing the effect of competition distance on sales...")
        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            data=self.df,
            x="CompetitionDistance",
            y="Sales",
            hue="StoreType",
            palette="coolwarm",
        )
        plt.title("Sales vs Competition Distance")
        plt.xlabel("Competition Distance (meters)")
        plt.ylabel("Sales")
        plt.grid(True)
        plt.legend(title="Store Type")
        plt.show()

    def competition_sales_distribution(self):
        """Visualize sales distribution for stores based on competition distance proximity."""
        print("Visualizing sales distribution by competition distance groups...")
        self.df["CompetitionProximity"] = pd.cut(
            self.df["CompetitionDistance"],
            bins=[0, 1000, 5000, 10000, 50000],
            labels=["<1km", "1-5km", "5-10km", ">10km"],
            include_lowest=True,
        )

        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=self.df, x="CompetitionProximity", y="Sales", palette="viridis"
        )
        plt.title("Sales Distribution by Competition Proximity")
        plt.xlabel("Competition Proximity")
        plt.ylabel("Sales")
        plt.grid(True)
        plt.show()

    def sales_vs_distance_trend(self):
        """Plot the trend of average sales with increasing competition distance."""
        print(
            "Analyzing trends of average sales with increasing competition distance..."
        )
        self.df["DistanceBin"] = pd.cut(
            self.df["CompetitionDistance"],
            bins=[0, 1000, 5000, 10000, 50000, float("inf")],
            labels=["<1km", "1-5km", "5-10km", "10-50km", ">50km"],
        )

        avg_sales = self.df.groupby("DistanceBin")["Sales"].mean().reset_index()

        plt.figure(figsize=(8, 5))
        sns.barplot(data=avg_sales, x="DistanceBin", y="Sales", palette="magma")
        plt.title("Average Sales vs Competition Distance")
        plt.xlabel("Competition Distance")
        plt.ylabel("Average Sales")
        plt.grid(True)
        plt.show()
