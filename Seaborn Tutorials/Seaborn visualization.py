## SEABORN Tutorials

"""
Distribution plots
1. Displot
2. Joinplot
3. Pairplot

"""

# Partice the problem with IRIS dataset

"""f1- dataset is uni-variate
f1,f2 are bi-variate
 more than two feature three dimension """
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")
print(df.head())

"""
Correlation with Heatmap

A Correlation heatmap uses colored cells, typically in a monochromatic scales, to show a 2D correlation matrix
(table) between two discrete dimension types, Its is very important in feature selection
"""

print(df.dtypes)

# to find correlation the feature value should be float or numerical only
# print(df.corr(numeric_only=True))
#
# import matplotlib.pyplot as plt
# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
# plt.title("Correlation Heatmap - Tips Dataset")
# plt.show()

"""
here the correlation was very good compare with itself  =1 
in case tip vs total_bill are good correlation approximate  = 0.68 if total_ bill increase tips also increase
in low case tip vs size are less correlation = 0.49

"""

## Uni variate Analysis

# sns.jointplot(x = "tip", y= "total_bill", data = df, kind = "hex")
# plt.show()

# sns.jointplot(x = "tip", y= "total_bill", data = df, kind = "reg")
# plt.show()

## Pair Plot

"""
A Pair plot is also known as a scatterplot, in which one variable in the same data row is matched with another 
variables value, like this: Paris plot are just eleboration on this, showing all variable paried with all the other 
variable
"""

# sns.pairplot(df)
# plt.savefig("pair plot.png")
# plt.show()
#
# sns.pairplot(df, hue= "sex")
# plt.savefig("tips plot based on gender.png")
# plt.title("Total bills and tips are based on Gender")
# plt.show()

# sns.pairplot(df, hue= "smoker")
# plt.savefig("tips based on gender.png")
# plt.title("Total bills and tips are based on smokers")
# plt.show()

print(df["smoker"].value_counts())

## displot

# kde = kernel density extension
sns.distplot(df["tip"],kde=False, bins= 10)
plt.show()
