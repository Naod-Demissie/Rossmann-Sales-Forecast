import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class StoreAnalysis:
    def __init__(self, df):
        """
        Initialize the StoreAnalysis class.

        Parameters:
        df (pd.DataFrame): DataFrame containing store-related data.
        """
        self.df = df

    def sales_distribution(self):
        """Analyze sales distribution across different stores."""
        print("Analyzing the overall distribution of sales across all stores...")
        plt.figure(figsize=(8, 5))
        sns.histplot(data=self.df, x="Sales", bins=50, kde=True, color="blue")
        plt.title("Sales Distribution Across Stores")
        plt.xlabel("Sales")
        plt.ylabel("Frequency")
        plt.show()

    def top_and_bottom_stores(self, top_n=10):
        """
        Identify top-performing and underperforming stores.

        Parameters:
        top_n (int): Number of top and bottom stores to display.
        """
        print(f"Identifying the top {top_n} and bottom {top_n} performing stores...")
        store_sales = self.df.groupby("Store")["Sales"].sum().sort_values()
        top_stores = store_sales.tail(top_n)
        bottom_stores = store_sales.head(top_n)

        plt.figure(figsize=(11, 5))

        # Plot for top-performing stores
        plt.subplot(1, 2, 1)
        top_stores.plot(
            kind="bar", color="#1f77b4", title="Top Performing Stores"
        )  # Blue
        plt.ylabel("Total Sales")
        plt.xlabel("Store")

        # Plot for underperforming stores
        plt.subplot(1, 2, 2)
        bottom_stores.plot(
            kind="bar", color="#ff7f0e", title="Underperforming Stores"
        )  # Orange
        plt.ylabel("Total Sales")
        plt.xlabel("Store")

        plt.tight_layout()
        plt.show()

    def store_type_performance(self):
        """Analyze sales and customer numbers by StoreType using subplots."""
        print(
            "Analyzing the performance of different store types in terms of sales and customer numbers..."
        )
        # Create a figure with 2 subplots (1 row, 2 columns)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

        # Plot sales distribution by StoreType
        sns.boxplot(data=self.df, x="StoreType", y="Sales", palette="Set2", ax=axes[0])
        axes[0].set_title("Sales Distribution by StoreType")
        axes[0].set_xlabel("StoreType")
        axes[0].set_ylabel("Sales")

        # Plot customer distribution by StoreType
        sns.boxplot(
            data=self.df, x="StoreType", y="Customers", palette="Set3", ax=axes[1]
        )
        axes[1].set_title("Customer Distribution by StoreType")
        axes[1].set_xlabel("StoreType")
        axes[1].set_ylabel("Customers")

        plt.tight_layout()
        plt.show()

    def compare_store_types(self):
        """Compare store types on key metrics using subplots."""
        print("Comparing store types on key metrics like sales and customer numbers...")

        fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)

        metrics = ["Sales", "Customers"]
        for i, metric in enumerate(metrics):
            sns.barplot(
                data=self.df,
                x="StoreType",
                y=metric,
                ci="sd",
                palette="muted",
                ax=axes[i],
            )
            axes[i].set_title(f"Average {metric} by StoreType")
            axes[i].set_xlabel("StoreType")
            axes[i].set_ylabel(metric)

        # Adjust layout for better spacing
        plt.tight_layout()
        plt.show()

    def assortment_performance(self):
        """Compare performance metrics between different assortments."""
        print("Comparing the performance metrics between different assortments...")
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=self.df, x="Assortment", y="Sales", palette="coolwarm")
        plt.title("Sales Distribution by Assortment")
        plt.xlabel("Assortment")
        plt.ylabel("Sales")
        plt.show()

    def holiday_impact_by_store_type(self):
        """Evaluate holiday impacts across different store types."""
        print("Analyzing holiday impacts across different store types...")

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))

        # Plot for Sales During Holidays
        sns.barplot(
            data=self.df,
            x="StoreType",
            y="Sales",
            hue="StateHoliday",
            palette="Set2",
            ci=None,
            ax=axes[0],
        )
        axes[0].set_title("Sales During Holidays by Store Type")
        axes[0].set_xlabel("Store Type")
        axes[0].set_ylabel("Average Sales")
        axes[0].legend(title="State Holiday")

        # Plot for Customer Numbers During Holidays
        sns.barplot(
            data=self.df,
            x="StoreType",
            y="Customers",
            hue="SchoolHoliday",
            palette="Set3",
            ci=None,
            ax=axes[1],
        )
        axes[1].set_title("Customer Numbers During Holidays by Store Type")
        axes[1].set_xlabel("Store Type")
        axes[1].set_ylabel("Average Customers")
        axes[1].legend(title="School Holiday")

        plt.tight_layout()
        plt.show()

    def holiday_impact_by_assortment(self):
        """Evaluate holiday impacts across different assortments."""
        print("Analyzing holiday impacts across different assortments...")

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))

        # Plot for Sales During Holidays
        sns.barplot(
            data=self.df,
            x="Assortment",
            y="Sales",
            hue="StateHoliday",
            palette="pastel",
            ci=None,
            ax=axes[0],
        )
        axes[0].set_title("Sales During Holidays by Assortment")
        axes[0].set_xlabel("Assortment Type")
        axes[0].set_ylabel("Average Sales")
        axes[0].legend(title="State Holiday")

        # Plot for Customer Numbers During Holidays
        sns.barplot(
            data=self.df,
            x="Assortment",
            y="Customers",
            hue="SchoolHoliday",
            palette="husl",
            ci=None,
            ax=axes[1],
        )
        axes[1].set_title("Customer Numbers During Holidays by Assortment")
        axes[1].set_xlabel("Assortment Type")
        axes[1].set_ylabel("Average Customers")
        axes[1].legend(title="School Holiday")

        plt.tight_layout()
        plt.show()
