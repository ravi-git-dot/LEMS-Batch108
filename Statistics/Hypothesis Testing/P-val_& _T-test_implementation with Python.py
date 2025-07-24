'''
## T Test
A t-test is a type of inferential statistic which is used to determine if there is a significant difference between the
means of two groups which may be related in certain features

T-test has 2 types : 1. one sampled t-test 2. two-sampled t-test.

One-sample T-test with Python
The test will tell us whether means of the sample and the population are different

'''


ages=[10,20,35,50,28,40,55,18,16,55,30,25,43,18,30,28,14,24,16,17,32,35,26,27,65,18,43,23,21,20,19,70]
print(len(ages))

import numpy as np
ages_mean = np.mean(ages)
print(ages_mean)

## let take a sample

Sample_size = 10
age_sample = np.random.choice(ages, Sample_size, replace=False)
print(age_sample)

from scipy.stats import ttest_1samp
ttest, p_value = ttest_1samp(ages, 30)
print(p_value)

# What is a p-value?
# The p-value tells you the probability that the results you got happened by chance
# (assuming there’s actually no real difference between the groups).
#
# A small p-value means the result is unlikely due to chance → the difference is likely real.
#
# A large p-value means the result could have happened by chance → the difference is not significant.

if p_value <0.05: # alpha value is 0.05 or 50%
    print("We are rejected Null Hypothesis")
else:
    print("We are accepted Null Hypothesis")

'''__________________________________________________________________________________________________________________'''
## Some more examples

import numpy as np
import pandas as pd
import scipy.stats as stats
import math

np.random.seed(6)
school_ages=stats.poisson.rvs(loc=18,mu=35,size=1500) #left most_val, mean, size
classA_ages=stats.poisson.rvs(loc=18,mu=30,size=60)   #left most_val, mean, size


print(classA_ages.mean())
_,p_value=stats.ttest_1samp(a=classA_ages,popmean=school_ages.mean())
print(p_value)

print(school_ages.mean())

if p_value < 0.05:  # alpha value is 0.05 or 50%
    print("We are rejected Null Hypothesis")
else:
    print("We are accepted Null Hypothesis")

'''__________________________________________________________________________________________________________________'''


'''
##Two-sample T-test With Python
The Independent Samples t Test or 2-sample t-test compares the means of two independent groups in order to determine 
whether there is statistical evidence that the associated population means are significantly different. 
The Independent Samples t Test is a parametric test. This test is also known as: Independent t Test
'''


np.random.seed(12)
ClassB_ages=stats.poisson.rvs(loc=18,mu=33,size=60)
print(ClassB_ages.mean())

_,p_value=stats.ttest_ind(a=classA_ages,b=ClassB_ages,equal_var=False)
print(p_value)

if p_value < 0.05:    # alpha value is 0.05 or 5%
    print(" we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")

'''__________________________________________________________________________________________________________________'''

## Paired T-test With Python
# When you want to check how different samples from the same group are, you can go for a paired T-test


weight1 = [25, 30, 28, 35, 28, 34, 26, 29, 30, 26, 28, 32, 31, 30, 45]
weight2 = weight1 + stats.norm.rvs(scale=5, loc=-1.25, size=15)

print("Weight1",weight1)
print('Weight2',weight2)

weight_df=pd.DataFrame({"weight_10":np.array(weight1),
                         "weight_20":np.array(weight2),
                       "weight_change":np.array(weight2)-np.array(weight1)})

print(weight_df)

_,p_value=stats.ttest_rel(a=weight1,b=weight2)
print(p_value)

if p_value < 0.05:    # alpha value is 0.05 or 5%
    print(" we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")

'''_________________________________________________________________________________________________________________'''

# Correlation

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

df =sns.load_dataset("iris")
print(df.shape)

sns.pairplot(df,hue="species")
plt.show()