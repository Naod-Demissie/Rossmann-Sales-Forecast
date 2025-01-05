import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class TrendAnalysis:
    def __init__(self, df):
        """
        Initialize with a dataframe.
        Parameters:
        - df: DataFrame with columns ['Date', 'Sales', 'Customers', 'DayOfWeek', 'StateHoliday', 'SchoolHoliday', 'Promo']
        """
        self.df = df.copy()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df["Year"] = self.df["Date"].dt.year
        self.df["Month"] = self.df["Date"].dt.month
        self.df["Week"] = self.df["Date"].dt.isocalendar().week

    def analyze_growth_rate(self):
        """Calculate and visualize daily sales growth rates."""
        daily_sales = self.df.groupby("Date")["Sales"].sum()
        growth_rate = daily_sales.pct_change() * 100

        plt.figure(figsize=(14, 6))
        plt.plot(
            growth_rate, label="Daily Sales Growth Rate (%)", color="orange", alpha=0.8
        )
        plt.axhline(0, color="black", linestyle="--", linewidth=0.7)
        plt.title("Daily Sales Growth Rate", fontsize=14)
        plt.xlabel("Date")
        plt.ylabel("Growth Rate (%)")
        plt.grid()
        plt.legend()
        plt.show()

    def weekly_sales_heatmap(self):
        """Create a heatmap of weekly sales trends."""
        weekly_sales = self.df.groupby(["Year", "Week"])["Sales"].mean().unstack()

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            weekly_sales,
            cmap="coolwarm",
            annot=False,
            cbar_kws={"label": "Average Sales"},
        )
        plt.title("Weekly Sales Heatmap", fontsize=14)
        plt.xlabel("Week of Year")
        plt.ylabel("Year")
        plt.show()

    def analyze_customer_trends(self):
        """Analyze customer trends over time."""
        daily_customers = self.df.groupby("Date")["Customers"].sum()

        plt.figure(figsize=(14, 6))
        plt.plot(daily_customers, label="Daily Customers", alpha=0.7, color="blue")
        plt.title("Daily Customer Trends", fontsize=14)
        plt.ylabel("Total Customers")
        plt.legend()
        plt.grid()

        xticks = daily_customers.index[:: len(daily_customers) // 10]
        plt.xticks(ticks=xticks, labels=xticks.strftime("%Y-%m-%d"), rotation=45)

        plt.tight_layout()
        plt.show()
