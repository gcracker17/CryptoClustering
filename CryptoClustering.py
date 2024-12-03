# %%
# Import required libraries and dependencies
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# %%
# Load the data into a Pandas DataFrame and make the index the "coin_id" column.
market_data_df = pd.read_csv("Resources/crypto_market_data.csv", index_col="coin_id")

# Display sample data
market_data_df.head(10)

# %%
# Generate summary statistics
market_data_df.describe()

# %% [markdown]
# ### Prepare the Data

# %%
# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
scaler = StandardScaler()
normalized_data = scaler.fit_transform(market_data_df)
normalized_df = pd.DataFrame(normalized_data, columns=market_data_df.columns)
normalized_df.head()

# %%
# Create a DataFrame with the scaled data
scaled_normalized_df = normalized_df

# Copy the crypto names from the original data
crypto_names = scaled_normalized_df['coinid']

# Set the coinid column as index
crypto_names.set_index('coinid', inplace=True)

# Display sample data
crypto_names.head()

# %% [markdown]
# ### Find the Best Value for k Using the Original Scaled DataFrame.

# %%
# Create a list with the number of k-values to try
# Use a range from 1 to 11


# Create an empty list to store the inertia values


# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using the scaled DataFrame
# 3. Append the model.inertia_ to the inertia list


# Create a dictionary with the data to plot the Elbow curve


# Create a DataFrame with the data to plot the Elbow curve


# Display the DataFrame


# %%
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.


# %% [markdown]
# #### Answer the following question: 
# **Question:** What is the best value for `k`?
# 
# **Answer:**

# %% [markdown]
# ### Cluster Cryptocurrencies with K-means Using the Original Scaled Data.

# %%
# Initialize the K-Means model using the best value for k


# %%
# Fit the K-Means model using the scaled data


# %%
# Predict the clusters to group the cryptocurrencies using the scaled data


# View the resulting array of cluster values.


# %%
# Create a copy of the DataFrame


# %%
# Add a new column to the DataFrame with the predicted clusters


# Display sample data


# %%
# Create a scatter plot using Pandas plot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`.
# Use "rainbow" for the color to better visualize the data.


# %% [markdown]
# ### Optimize Clusters with Principal Component Analysis.

# %%
# Create a PCA model instance and set `n_components=3`.


# %%
# Use the PCA model with `fit_transform` on the original scaled DataFrame to reduce to three principal components.


# View the first five rows of the DataFrame. 


# %%
# Retrieve the explained variance to determine how much information  can be attributed to each principal component.


# %% [markdown]
# #### Answer the following question: 
# 
# **Question:** What is the total explained variance of the three principal components?
# 
# **Answer:** 

# %%
# Create a new DataFrame with the PCA data.
# Note: The code for this step is provided for you

# Creating a DataFrame with the PCA data


# Copy the crypto names from the original data


# Set the coinid column as index


# Display sample data


# %% [markdown]
# ### Find the Best Value for k Using the PCA Data

# %%
# Create a list with the number of k-values to try
# Use a range from 1 to 11


# Create an empty list to store the inertia values


# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using PCA DataFrame.
# 3. Append the model.inertia_ to the inertia list


# Create a dictionary with the data to plot the Elbow curve


# Create a DataFrame with the data to plot the Elbow curve


# Display the DataFrame


# %%
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.


# %% [markdown]
# #### Answer the following questions: 
# * **Question:** What is the best value for `k` when using the PCA data?
# 
#   * **Answer:** 
# 
# 
# * **Question:** Does it differ from the best k value found using the original data?
# 
#   * **Answer:** 

# %% [markdown]
# ### Cluster Cryptocurrencies with K-means Using the PCA Data

# %%
# Initialize the K-Means model using the best value for k


# %%
# Fit the K-Means model using the PCA data


# %%
# Predict the clusters to group the cryptocurrencies using the PCA data


# Print the resulting array of cluster values.


# %%
# Create a copy of the DataFrame with the PCA data


# Add a new column to the DataFrame with the predicted clusters


# Display sample data


# %%
# Create a scatter plot using hvPlot by setting `x="PCA1"` and `y="PCA2"`. 


# %% [markdown]
# ### Determine the Weights of Each Feature on each Principal Component

# %%
# Use the columns from the original scaled DataFrame as the index.


# %% [markdown]
# #### Answer the following question: 
# 
# * **Question:** Which features have the strongest positive or negative influence on each component? 
#  
# * **Answer:** 
#     

# %%



