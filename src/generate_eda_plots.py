import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def generate_eda_visualizations(data_path: str = "data/processed/pune_house_data_cleaned.csv",
                                output_dir: str = "reports/figures"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    
    sns.set_theme(style="whitegrid", palette="muted")
    
    # -------------------------------------------------------------
    # Plot 1: Price Distribution (Right Skewness)
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    sns.histplot(df['price'], bins=60, kde=True, color='#2563eb')
    plt.title('Plot 1: Distribution of Property Prices (Lakhs INR)', fontsize=13, fontweight='bold')
    plt.xlabel('Price (₹ Lakhs)', fontsize=11)
    plt.ylabel('Listing Count', fontsize=11)
    plt.axvline(df['price'].median(), color='red', linestyle='--', label=f'Median Price: ₹{df["price"].median():.1f} L')
    plt.axvline(df['price'].mean(), color='orange', linestyle='-', label=f'Mean Price: ₹{df["price"].mean():.1f} L')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_price_distribution.png"), dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot 2: Boxplot of Price by BHK
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    bhk_subset = df[df['bhk'].between(1, 5)]
    sns.boxplot(x='bhk', y='price', data=bhk_subset, palette='Blues_r', showfliers=False)
    plt.title('Plot 2: Property Price Distribution by BHK (1 to 5 BHK)', fontsize=13, fontweight='bold')
    plt.xlabel('BHK (Number of Bedrooms)', fontsize=11)
    plt.ylabel('Price (₹ Lakhs)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_price_by_bhk_boxplot.png"), dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot 3: Scatter Plot of Total Sqft vs Price
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    scatter = sns.scatterplot(
        data=df[df['bhk'].between(1, 4)],
        x='total_sqft', y='price', hue='bhk',
        palette='viridis', alpha=0.6, s=30
    )
    plt.title('Plot 3: Total Square Footage vs Price (Colored by BHK)', fontsize=13, fontweight='bold')
    plt.xlabel('Total Area (Sq. Ft.)', fontsize=11)
    plt.ylabel('Price (₹ Lakhs)', fontsize=11)
    plt.legend(title='BHK')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_sqft_vs_price_scatter.png"), dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot 4: Top 15 Localities by Median Price Per SqFt
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    top_localities = (
        df[df['site_location'] != 'other']
        .groupby('site_location')['price_per_sqft']
        .median()
        .sort_values(ascending=False)
        .head(15)
    )
    sns.barplot(x=top_localities.values, y=top_localities.index, palette='crest')
    plt.title('Plot 4: Top 15 Localities by Median Price per Sq. Ft. (INR)', fontsize=13, fontweight='bold')
    plt.xlabel('Median Price per Sq. Ft. (₹)', fontsize=11)
    plt.ylabel('Locality', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_top_localities_price_per_sqft.png"), dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot 5: Correlation Heatmap
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    numeric_cols = ['total_sqft', 'bhk', 'bath', 'balcony', 'price', 'price_per_sqft']
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True, linewidths=0.5)
    plt.title('Plot 5: Feature Correlation Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_correlation_heatmap.png"), dpi=150)
    plt.close()
    
    print(f"All 5 EDA visualizations saved successfully in '{output_dir}/'")

if __name__ == "__main__":
    generate_eda_visualizations()
