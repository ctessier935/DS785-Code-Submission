# -*- coding: utf-8 -*-
"""
@author: Chase Tessier
"""

'''
Code gets basic statistics and correlation information to support presentation 3.
'''

import pandas as pd
from scipy.stats import chi2_contingency

# Determining correlation between Credit Risk, Current Loan Amount, and Interest Rate.
df = pd.read_excel("C:\\Users\\chase\\Desktop\\Data Science\\DS785 - Capstone\\Capstone Project - Loan Data.xlsx")
df['CREDITRISKCODE'] = df['CREDITRISKCODE'].astype(str).str.strip()

df['CREDITRISKCODE'].replace('S',10,inplace=True)

df_corr = df[['CREDITRISKCODE', 'CURRENTLOANAMOUNT', 'INTERESTRATE']].corr()


print(df_corr)

##############################

# Chi Square and p value correlation between Credit Risk and Loan Type.
df = pd.read_excel("C:\\Users\\chase\\Desktop\\Data Science\\DS785 - Capstone\\Capstone Project - Loan Data.xlsx")
df['CREDITRISKCODE'] = df['CREDITRISKCODE'].astype(str).str.strip()

# Create a contingency table
contingency_table = pd.crosstab(df['CREDITRISKCODE'], df['PROCAPPLCODE'])
print("Contingency Table:")
print(contingency_table)

# Perform the Chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"\nChi-square statistic: {chi2}")
print(f"P-value: {p_value}")
print(f"Degrees of freedom: {dof}")
print(f"Expected frequencies:\n{expected}")

# Interpret the p-value
alpha = 0.05
if p_value < alpha:
    print("\nReject the null hypothesis: There is a significant association between Credit Risk and Loan Type.")
else:
    print("\nFail to reject the null hypothesis: There is no significant association between Credit Risk and Loan Type.")