import pandas as pd

# Load dataset
df = pd.read_csv("train.csv")

# show first 5 rows
print(df.head())
# Basic information about the dataset
print("\nDataset Information:")
print(df.info())

# Number of rows and columns
print("\nShape of dataset:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns)
# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())
# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())
# Data Cleaning

# Fill missing Age values with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin column because it has many missing values
df = df.drop("Cabin", axis=1)

# Check missing values again
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
# Data Cleaning

# Fill missing Age values with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin column because it has many missing values
# df= df.drope("cabin",axis=1)

# Check missing values again
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
# Survival Analysis

print("\nSurvival Count:")
print(df["Survived"].value_counts())

print("\nSurvival Percentage:")
print(df["Survived"].value_counts(normalize=True) * 100)

print("\nSurvival by Gender:")
print(df.groupby("Sex")["Survived"].mean() * 100)

print("\nSurvival by Passenger Class:")
print(df.groupby("Pclass")["Survived"].mean() * 100)
print("\nAverage Age:")
print(df["Age"].mean())

print("\nAverage Fare:")
print(df["Fare"].mean())

print("\nAverage Fare by Passenger Class:")
print(df.groupby("Pclass")["Fare"].mean())

print("\nAverage Age by Passenger Class:")
print(df.groupby("Pclass")["Age"].mean())
import matplotlib.pyplot as plt
import seaborn as sns

# Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()
# Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived")
plt.show()
# Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived")
plt.show()
# Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived")
plt.show()
# Correlation Heatmap
plt.figure(figsize=(8, 5))

numeric_df = df.select_dtypes(include="number")

sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.show()
# TASK 3 - DATA VISUALIZATION

# 1. Survival Rate by Gender
plt.figure(figsize=(6, 4))
sns.barplot(x="Sex", y="Survived", data=df)
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.show()


# 2. Survival Rate by Passenger Class
plt.figure(figsize=(6, 4))
sns.barplot(x="Pclass", y="Survived", data=df)
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.show()


# 3. Fare Distribution
plt.figure(figsize=(7, 4))
sns.histplot(df["Fare"], bins=30, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.show()