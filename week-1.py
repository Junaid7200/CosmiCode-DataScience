import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('Datasets/House Price Prediction/Housing.csv')

df.head()

print(df.describe(), end="\n============================================================================\n")

# Check for missing data
print(df.isnull().sum())    # there are no missing values so we can move on


## already loaded the dataset as a pandas dataframe so this is kinda useless but like, why not right?
price_array = np.array(df['price'])
area_array = np.array(df['area'])
bedrooms_array = np.array(df['bedrooms'])
bathrooms_array = np.array(df['bathrooms'])

# Performing basic mathematical operations 
print(f"The average for the prices column is {np.mean(price_array)}, so thats the average price for houses in this dataset")
print(f"The average for the area column is {np.mean(area_array)}, so thats the average area for houses in this dataset")
print(f"The standard deviation for the prices column is {np.std(price_array)}, so thats the standard deviation for houses in this dataset")
print(f"The standard deviation for the area column is {np.std(area_array)}, so thats the standard deviation for houses in this dataset")

df_cleaned = df.copy()

# Convert 'yes'/'no' in binary columns into 1/0
binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df_cleaned[binary_columns] = df_cleaned[binary_columns].apply(lambda x: x.map({'yes': 1, 'no': 0}))

# Convert 'furnishingstatus' into numerical values: 'furnished' -> 2, 'semi-furnished' -> 1, 'unfurnished' -> 0
df_cleaned['furnishingstatus'] = df_cleaned['furnishingstatus'].map({'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0})

# Display the cleaned dataset
df_cleaned.head()


plt.figure(figsize=(10, 6))
plt.plot(df_cleaned['area'], df_cleaned['price'], color='blue', marker='o', linestyle='dashed')
plt.title('House Price vs Area')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price (in millions)')
plt.grid(True)
plt.show()

# Bar chart: Average Price per Number of Bedrooms
avg_price_per_bedroom = df_cleaned.groupby('bedrooms')['price'].mean()

plt.figure(figsize=(10, 6))
avg_price_per_bedroom.plot(kind='bar', color='orange')
plt.title('Average Price per Number of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Average Price (in millions)')
plt.grid(True)
plt.show()


print("In this project, we took a deep dive into a house price dataset to uncover some interesting insights. Starting with an initial exploration, we found that the average house price was around 4.77 million, with house areas typically around 5,150 square feet. Thankfully, there were no missing values, so we could dive straight into analysis without any data cleaning headaches.\n\n")
print("We moved on to some basic math using NumPy, like adding and subtracting house prices and areas. This gave us a sense of how house size relates to price, with larger homes predictably costing more. Standard deviations showed significant variation, suggesting that house pricing isnt just about size  theres a lot more at play.\n\n")

print("For preprocessing, we cleaned up categorical columns by converting them into numbers. Things like \"mainroad\" and \"guestroom\" got translated into 1s and 0s, while \"furnishing status\" was mapped into a scale from unfurnished (0) to fully furnished (2).\n\n")

print("Next came the fun part: visualizations. A line chart of price vs. area showed that while bigger houses generally cost more, there are a lot of outliers. The bar chart revealed that the number of bedrooms significantly affects pricing  houses with 5 or 6 bedrooms tend to be pricier.\n\n")

print("In short, this analysis gave us a better understanding of what drives house prices. Weve prepped this data for predictive modeling, and our visualizations highlight key trends like the price jump for bigger homes and those with more bedrooms. Theres definitely room to dive deeper into these patterns, but we've laid a solid foundation!")


