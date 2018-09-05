#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 4 13:57:10 2018
Survival Analysis Recovery Data
@author: Wanlu
"""
#In terminal I installed the lifelines package: pip install 'lifelines'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
#from lifelines.statistics import multivariate_logrank_test
from lifelines import CoxPHFitter

#Load dataframe
recovery_df = pd.read_csv('recover_final2.csv')
print(recovery_df.iloc[1:10, :])
#print(recovery_df.head())

#Data cleaning
#Sex
#gender = {'FEMALE': 1,
#          'MALE': 0
#          }
#recovery_df['Sex'] = recovery_df.Sex.replace(gender, regex=True)
recovery_df['Sex'] = recovery_df['Sex'].map({'FEMALE': 1, 'MALE': 0})
print(recovery_df.iloc[1:10, :])
#Race
grouped_by_race = recovery_df.groupby('Race')
grouped_by_race = grouped_by_race['Race'].aggregate('count')
print(grouped_by_race)
recovery_df.loc[recovery_df['Race'].isin(['WHITE', 'BLACK']) == False, 'Race'] = 2
recovery_df.loc[recovery_df.Race == 'WHITE', 'Race'] = 0
recovery_df.loc[recovery_df.Race == 'BLACK', 'Race'] = 1
print(recovery_df.iloc[1:10, :])
#LOS and Age
recovery_df.plot(y = 'LOS', kind = 'hist', color = 'DarkOrange')
recovery_df.plot(y = 'Admit_Age', kind = 'hist', color = 'DarkGreen')
#Description
recovery_df.iloc[:,1:8].describe()

#Create subsets of dataframe
print(recovery_df.shape[0])
recovery_df = recovery_df[recovery_df.month.notnull()]
t_group = recovery_df.loc[recovery_df['Recover_or_not'] == 1]
i_group = recovery_df.loc[recovery_df['Recover_or_not'] == 0]
print(recovery_df.shape[0], t_group.shape[0], i_group.shape[0])
#recovery_df.Recover_or_not.unique()
#recovery_df.Recover_or_not.value_counts()

#Kaplan-Meier Curves
#Treatment group
kmf_t = KaplanMeierFitter()
kmf_t.fit(durations = t_group.month, event_observed = t_group.death)
print(kmf_t.event_table)
print(kmf_t.survival_function_)
#Set plotting aesthetics
sns.set(palette = 'bright', font_scale = 1.35, 
        rc = {'figure.figsize': (8, 6), 'axes.facecolor': '.92'})
kmf_t.plot()
plt.title('The Kaplan-Meier Estimate for Recovery Group')
plt.ylabel('Probability a Patient is Still Alive')
plt.show()
#Control group
kmf_i = KaplanMeierFitter()
kmf_i.fit(durations = i_group.month, event_observed = i_group.death)
print(kmf_i.event_table)
print(kmf_i.survival_function_)
#Set plotting aesthetics
sns.set(palette = 'colorblind', font_scale = 1.35, 
        rc = {'figure.figsize': (8, 6), 'axes.facecolor': '.92'})
kmf_i.plot()
plt.title('The Kaplan-Meier Estimate for Impairment Group')
plt.ylabel('Probability a Patient is Still Alive')
plt.show()
#Difference
print(kmf_t.subtract(kmf_i))

#Kaplan-Meier Curves 2.0
#Set plotting aesthetics
sns.set(palette = 'colorblind', font_scale = 1.35, 
        rc = {'figure.figsize': (8, 6), 'axes.facecolor': '.92'})
kmf_by_group = KaplanMeierFitter()
duration = recovery_df.month
observed = recovery_df.death
groups = [1, 0]
titles = ['Recovery', 'Impairment']
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 3))
for grp, ax, til in zip(groups, axes.flatten(), titles):
    idx = recovery_df.Recover_or_not == grp
    kmf_by_group.fit(duration[idx], observed[idx])
    kmf_by_group.plot(ax = ax, legend = False)
#    ax.annotate('Median = {:.0f} months'.format(kmf_by_group.median_), xy = (.47, .85), 
#                xycoords = "axes fraction")
    ax.set_xlabel('')
    ax.set_title(til, fontsize = 12)
    ax.set_xlim(0, recovery_df['month'].max())
    ax.set_ylim(0.5, 1)
fig.tight_layout()
fig.text(0.5, -0.01, 'Time(Months)', ha = 'center', fontsize = 12)
fig.text(-0.01, 0.5, "Probability a Patient is Still Alive", 
         va = 'center', rotation = 'vertical', fontsize = 12)
fig.suptitle('Survival Curves for Recovery and Impairment Groups',
             fontsize = 14)
#fig.subplots_adjust(top = 0)
plt.show()

#Kaplan-Meier Curves 3.0
kmf = KaplanMeierFitter()
ax = plt.subplot(111)
t = np.linspace(0, 50, 51)
recover = (recovery_df["Recover_or_not"] == 1)
kmf.fit(recovery_df['month'][recover],
        event_observed = recovery_df['death'][recover],
        timeline = t, label = 'Recovery')
kmf.plot(ax = ax, ci_force_lines = True)
kmf.fit(recovery_df['month'][~recover],
        event_observed = recovery_df['death'][~recover],
        timeline = t, label = 'Impairment')
kmf.plot(ax = ax, ci_force_lines = True)
plt.ylim(0.5, 1)
plt.title('Survival Curves for Recovery and Impairment Groups')

#Log Rank Test
results = logrank_test(recovery_df['month'][recover], recovery_df['month'][~recover],
                       recovery_df['death'][recover], recovery_df['death'][~recover],
                       alpha = .95)
print(results.print_summary())
#results2 = multivariate_logrank_test(recovery_df['month'], recovery_df['Recover_or_not'], recovery_df['death'])
#print(results2.print_summary())

#Cox Regression
cph = CoxPHFitter()
cph.fit(recovery_df.iloc[:, 1:8], duration_col = 'month', event_col = 'death', show_progress = True)
print(cph.print_summary())
