# Titanic-Survival-Analysis-Data-Analytics
Data cleaning, processing and Data Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df=pd.read_csv("/content/titanic.csv",encoding="1252")
df.head(12)
df.shape
df.tail(12)
df.describe ()
df.isnull().sum()
from IPython.display import display ,HTML
def show_all (df):
  with pd.option_context("display.max_rows", None,"display.max_columns", None):
    display(HTML(df.to_html()))
    plt.figure(figsize= (20,5))
sns.boxplot (data=df)
plt.title ("Overview of OT")
df.info()
df.ndim
df.columns
df['PassengerId'].describe()
plt.title('Null values Visualization:')
sns.heatmap(df.isnull(),cmap='CMRmap_r')
df.plot(kind='box', subplots= True, figsize=(25,5))
df.drop_duplicates(keep='last' ,inplace=True)
df.plot(kind='line',subplots=True,figsize=(25,15))
plt.title(label="How our dataset looks like : visualization",loc ='center',color='maroon')
sns.boxplot(data=df,x=df['Fare'])
plt.title('Box Plot')
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

# Define the upper and lower bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df_no_outliers = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]

print(f"Original shape: {df.shape}")
print(f"Shape after removing outliers: {df_no_outliers.shape}")
sns.boxplot(data=df_no_outliers,x=df_no_outliers['Fare'])
plt.title('Box Plot')
#Set the style for all visualization
sns.set(style="whitegrid")
plt.figure(figsize=(15, 10))

#1. Distribution of People Survived in titanic (genderwise)
plt.subplot(2, 2, 1)
sns.histplot(data=df, x='Survived', hue='Sex', multiple='stack')
plt.title('Distribution of People Survived in titanic (genderwise)')
plt.xlabel('Survived')
plt.ylabel('Count')

plt.subplot(2, 2, 2)
sns.boxplot(data=df, x='Age')
plt.title('Box Plot of Age vs Survived')
plt.xlabel('Cabin')

plt.tight_layout()
plt.show()
