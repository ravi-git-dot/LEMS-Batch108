## Chi-Square Test
'''
The test is applied when you have two categorical variables from a single population. It is used to determine whether
there is a significant association between the two variables.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

df =sns.load_dataset('tips')
print(df.head())

df_table = pd.crosstab(df['sex'], df['smoker'])
print(df_table)

val=stats.chi2_contingency(df_table)
print(val)


#Observed Values
Observed_Values = df_table.values
print("Observed Values :-\n",Observed_Values)

Expected_Values=val[3]

no_of_rows=len(df_table.iloc[0:2,0])
no_of_columns=len(df_table.iloc[0,0:2])
ddof=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",ddof)
alpha = 0.05

from scipy.stats import chi2

chi_square = sum([(o - e) ** 2. / e for o, e in zip(Observed_Values, Expected_Values)])
chi_square_statistic = chi_square[0] + chi_square[1]

print("chi-square statistic:-",chi_square_statistic)


critical_value=chi2.ppf(q=1-alpha,df=ddof)
print('critical_value:',critical_value)

# p-value
p_value = 1 - chi2.cdf(x=chi_square_statistic, df=ddof)
print('p-value:', p_value)
print('Significance level: ', alpha)
print('Degree of Freedom: ', ddof)
print('p-value:', p_value)

if chi_square_statistic >= critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")

if p_value <= alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")