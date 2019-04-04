#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime


# In[51]:


# Loading the csv dataset into the df variable
df = pd.read_csv('crimes_2018.csv')


# In[52]:


# The number of rows and columns in the dataset; returns a tuple
df.shape


# In[53]:


df.describe()


# In[54]:


df.info()


# In[55]:


sns.set_style('darkgrid')


# In[57]:


# First 5 rows of our dataset
df.head()


# In[58]:


# The names of the features
df.columns


# In[59]:


# Number of distinct crimes in the city in 2018
crimes = df['Primary Type'].unique()
print("The Number of distinct crimes in Chicago in the year 2018:", len(crimes))
print()
print(crimes)


# In[60]:


# Let's count number of null entries
missing_values = list(df.isna().sum())
# missing values is a list of the number of missing values in each column
cols = list(df.columns)
for i in range(len(cols)):
    if (missing_values[i] == 0):
        cols[i]="Others"
d = dict(zip(cols, missing_values))


# In[61]:


x = list(d.keys())
y = list(d.values())
sns.barplot(x=x, y=y, palette="GnBu_d")
plt.xticks(rotation=90)
plt.title("Missing Values in the Dataset", fontdict = {'fontsize': 20})
plt.ylabel("Count of missing values", fontdict={'fontsize': 15})
plt.show()


# In[62]:


df = df.dropna()
df.info()


# In[63]:


print(round(262960 / 265698 * 100,2), "percentage of the data has been retained.")


# In[64]:


# Continuous Variables
cont = df._get_numeric_data().columns
print("The continuous variables are: ",list(cont))


# In[65]:


# Categorical Variables
print("The categorical variables are: ",list(set(df.columns) - set(cont)))


# In[66]:


df.head()


# In[67]:


df.columns


# <h4>Check for available plot styles</h4>

# In[70]:


# use plt.style.available to view all possible styles


# <h3>Top Crimes in Chicago in the year 2018</h3>

# In[72]:


# Set the style of the plot first
plt.style.use('bmh')
sns.set_context('notebook')

# Filter out the Top 5 crimes
top_5_crimes = df['Primary Type'].value_counts().sort_values(ascending=False).head()

temp = df.groupby('Primary Type', as_index=False).agg({"ID": "count"})
temp = temp.sort_values(by=['ID'], ascending=False).head()
sns.barplot(x='Primary Type', y='ID', data=temp, palette="Blues_d")

# Work on the aestehtic appeal of the plot
plt.title("Top 5 Crimes in Chicago\n", fontdict = {'fontsize': 20, 'color': '#bb0e14'})
plt.ylabel("COUNT OF CRIMES", fontdict = {'fontsize': 15})
plt.xlabel("TYPE OF CRIME", fontdict = {'fontsize': 15})
plt.xticks(rotation=90)
plt.show()
#plt.show()


# In[73]:


df.info()


# In[77]:


# Testing out the time and date conversion
t = df['Date'][20]
print(t)
s1 = t[:11] 
print(s1)
s2 = t[11:]
print(s2)

print(s2)
hr = s2[:2]
mins = s2[3:5]
sec = s2[6:8]
time_frame = s2[9:]
if(time_frame == 'PM'):
    if (int(hr) != 12):
        hr = str(int(hr) + 12)
else:
    if(int(hr) == 12):
        hr = '00'

print(hr, mins, sec)


# In[78]:


month = s1[:2]
date = s1[3:5]
year = s1[6:10]

final_date = datetime(int(year), int(month), int(date), int(hr), int(mins), int(sec))
print(final_date)


# <h4>Convert the date-time column to a known format</h4>

# In[79]:


# Time Conversion Function
def time_convert(date_time):
    s1 = date_time[:11]
    s2 = date_time[11:]
    
    month = s1[:2]
    date = s1[3:5]
    year = s1[6:10]
    
    hr = s2[:2]
    mins = s2[3:5]
    sec = s2[6:8]
    time_frame = s2[9:]
    if(time_frame == 'PM'):
        if (int(hr) != 12):
            hr = str(int(hr) + 12)
    else:
        if(int(hr) == 12):
            hr = '00'
    
    final_date = datetime(int(year), int(month), int(date), int(hr), int(mins), int(sec))
    return final_date
    
    


# In[80]:


# Using apply() of pandas to apply time_convert on every row of the Date column
df['Date'] = df['Date'].apply(time_convert)


# In[81]:


# Testing the status of AM or PM
df['Date'][0].strftime("%p")


# In[82]:


df['Date'].head(20)


# <h4>Create a new column "Month"</h4>

# In[83]:


def month(x):
    return x.strftime("%B")
df['Month'] = df['Date'].apply(month)


# In[84]:


df.head()[:5]


# <h4>The number of crimes in Chicago in 2018 as per month</h4>

# In[85]:


# Set plot style
plt.style.use('bmh')
sns.set_context('notebook')

# Code to plot
sns.countplot(y='Month', data=df, palette=["#bb0e14"], alpha=0.25)

# Aesthetic appeal of the plot 
plt.title("CRIMES PER MONTH\n", fontdict={'fontsize': 20, 'color': 'teal'}, weight="bold")
plt.ylabel("Month\n", fontdict={'fontsize': 15}, weight="bold")
plt.xlabel("\nNumber of Crimes", fontdict={'fontsize': 15}, weight="bold")
plt.show()


# <h4>Create a new column "Hour" [24 hour format]</h4>

# In[86]:


def hour(x):
    return x.strftime("%H")
df['Hour_Day'] = df['Date'].apply(hour)


# In[87]:


df.head()


# <h4>Distribution of crimes in Chicago in 2018 as per hour</h4>

# In[88]:


# Set plot style
plt.style.use('bmh')
sns.set_context('paper')

# Write code to plot
sns.countplot(x='Hour_Day', data=df, palette="Blues_d")

# Aesthetic appeal
plt.title("UNSAFEST HOURS\n", fontdict={'fontsize': 20, 'color': '#bb0e14'}, weight="bold")
plt.xlabel("\nHour in the Day", fontdict={'fontsize': 15}, weight='bold')
plt.ylabel("Number of Crimes\n", fontdict={'fontsize': 15}, weight="bold")
plt.show()


# In[89]:


df.head()


# In[90]:


df["Block"].value_counts()[:10]


# ### Crime Locations

# In[91]:


df.info()


# In[92]:


df.columns


# In[93]:


df.head()


# In[97]:


# Let's simulate the map of Chicago with the help of X and Y co-ordinates
sns.lmplot('X Coordinate', 
           'Y Coordinate',
           data=df,
           fit_reg=False, 
           hue="District",
           palette='colorblind',
           height=5,
           scatter_kws={"marker": "+", 
                        "s": 10})
ax = plt.gca()
ax.set_title("All Crime Distribution per District\n", fontdict={'fontsize': 15}, weight="bold")
plt.show()


# In[98]:


df.head()

# Let's simulate the map of Chicago with the help of X and Y co-ordinates
sns.lmplot('X Coordinate', 
           'Y Coordinate',
           data=df,
           fit_reg=False, 
           hue="Primary Type",
           palette='colorblind',
           height=5,
           ci=2,
           scatter_kws={"marker": "+", 
                        "s": 10})
ax = plt.gca()
ax.set_title("All Crime Distribution\n", fontdict={'fontsize': 15}, weight="bold")
plt.show()


# In[99]:


df.head()


# In[100]:


df.columns


# <h3>We need to do the following in our whole project</h3>
# <strong>Common Crimes in Chicago</strong>
# <ul><li>A graph depicting the most occurring offences in Chicago in 2018.</li>
#     <li>A comparison of the frequency of the top 5 most commonly committed offences</li>
# </ul>
# <strong>Crime vs Time</strong>
# <ul><li>Which hours are the most unsafest ?</li>
#     <li>What crimes are more prone to happen in the cover of darkness than in the morning ?</li>
#     <li>Is your house safe from a burglary during the day ?</li>
# </ul>
# <strong>Gang Related Offences</strong>
# <ul><li>What are gang related offences ?</li>
#     <li>Are there obvious patterns in distinguished gang activity in Chicago in 2018?</li>
# </ul>
# <strong>Crime vs Locations</strong>
# <ul><li>Which district is the safest to live in? Which district is "Sin-district" ?</li>
#     <li>Compare the top 5 committed offences with their regions of occurring. </li>
#     <li>You live in area X. You wish to see the crime stats in X in the year of 2018, especially of "Burglary". Visualize a time series graph to see the same.</li>
# </ul>
# <strong>Predict a Crime</strong>
# <ul><li>If you were a cop, which parts of Chicago would you deploy your extra forces on April 15, 2019 at 12:00 p.m.? (Based on the data you have)</li>
# </ul>    





