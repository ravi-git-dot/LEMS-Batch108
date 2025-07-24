##Anova Test(F-Test)
'''
The t-test works well when dealing with two groups, but sometimes we want to compare more than two groups at the same time.

For example, if we wanted to test whether petal_width age differs based on some categorical variable like species,
we have to compare the means of each level or group the variable


One Way F-test(Anova) :-
It tell whether two or more groups are similar or not based on their mean similarity and f-score.

Example : there are 3 different category of iris flowers and their petal width and need to
check whether all 3 group are similar or not
'''
import pandas as pd
import seaborn as sns
import scipy.stats as stats

df = sns.load_dataset('iris')
print(df.head())

df_anova = df[['petal_width','species']]

grps = pd.unique(df_anova.species.values)
print(grps)


d_data = {grp:df_anova['petal_width'][df_anova.species == grp] for grp in grps}
print(d_data)


F, p = stats.f_oneway(d_data['setosa'], d_data['versicolor'], d_data['virginica'])
print(p)


if p<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")