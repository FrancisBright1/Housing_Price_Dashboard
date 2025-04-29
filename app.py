import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page settings
st.set_page_config(page_title="Ames Housing Analysis", layout="wide")
st.title("Ames Housing Data Analysis Dashboard")

# Read the dataset
df = pd.read_csv('AmesHousing.csv')

# Display the dataset
st.subheader("Sample of the Dataset")
st.dataframe(df.head())

# Calculate correlations
corr_matrix = df.select_dtypes(include=['number']).corr()
top_features = corr_matrix['SalePrice'].abs().sort_values(ascending=False).head(20)

# Sort the correlation values with 'SalePrice'
correlation_with_saleprice = corr_matrix['SalePrice'].sort_values(ascending=False)

# Select features with a high correlation to 'SalePrice'
highly_correlated_features = correlation_with_saleprice[abs(correlation_with_saleprice) > 0.45].index

# Visualize the top 10 correlated features with SalePrice
st.subheader("Top 10 Correlated Features with SalePrice")

fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(y=correlation_with_saleprice.index[:10], x=correlation_with_saleprice.values[:10], palette="viridis", ax=ax1)
ax1.set_title("Top 10 Correlated Features with SalePrice")
ax1.set_xlabel("Correlation Coefficient")
ax1.set_ylabel("Feature")
st.pyplot(fig1)

# Visualize correlation heatmap
st.subheader("Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax2)
ax2.set_title("Correlation Heatmap of Numerical Features")
st.pyplot(fig2)

# Visualize the relationship between SalePrice and top features
st.subheader("SalePrice vs Top Features")

top_features_list = ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Total Bsmt SF', '1st Flr SF', 'Year Built']
fig3, axes = plt.subplots(2, 3, figsize=(18, 10))

for i, feature in enumerate(top_features_list):
    row, col = divmod(i, 3)
    sns.scatterplot(x=df[feature], y=df['SalePrice'], ax=axes[row, col], color='teal')
    axes[row, col].set_title(f"SalePrice vs {feature}")
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel("SalePrice")

plt.tight_layout()
st.pyplot(fig3)
