import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("reviews.csv")

# Display data
print("First 5 Reviews:")
print(df.head())

# Check dataset information
print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Sentiment count
print("\nSentiment Count:")
print(df["sentiment"].value_counts())

# Visualization
plt.figure(figsize=(7, 5))
sns.countplot(x="sentiment", data=df)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.show()