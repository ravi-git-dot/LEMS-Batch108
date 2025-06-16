## categorical plots

"""
Seaborn also helps us in doing analysis on categorical data points. In this section
we will discuss

1. Box plot
2. Violinplot
3. Countplot
4. Barplot

"""
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")
print(df.head())

## count plot

# sns.countplot(x ="sex", data=df, palette = "pastel")
# plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

# # Use built-in palette like "pastel", "Set2", "husl", etc.
# sns.countplot(x='day', data=df, palette='Set2')
# plt.title("Count of Bills per Day")
# plt.show()

# in y-axis direction
# sns.countplot(y= "sex", data=df)
# plt.show()

## Bar plot

# sns.barplot(x ="total_bill", y = "sex", data=df, palette="husl")
# plt.show()

# sns.barplot(x ="total_bill", y = "smoker", data=df, palette="husl")
# plt.show()

## box plot

# sns.boxplot(x ="sex", y= "total_bill", data=df, palette="Set2")
# plt.show()

# sns.boxplot(x ="day", y= "total_bill", data=df, palette="rainbow")
# plt.show()

# sns.boxplot(data=df, orient='v')
# plt.show()

# categorical my data based on some other categories

# sns.boxplot(x ="total_bill", y= "day", hue= "smoker", data=df)
# plt.show()

# sns.boxplot(x ="total_bill", y= "day", hue= "sex", data=df)
# plt.show()



## Violin Plot

"""
Violin plot helps to see both the distribution of data in terms of kernel density estimated 
and the box plot
"""

# sns.violinplot(x ="total_bill", y= "day", data=df, palette="rainbow")
# plt.show()

sns.violinplot(x ="total_bill", y= "sex", data=df, palette="rainbow")
plt.show()



