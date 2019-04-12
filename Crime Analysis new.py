#!/usr/bin/env python
# coding: utf-8

# <h1>Chicago Crime Analysis - Using Data Science to understand crime in Chicago in the year 2018</h1>
# <p>Open Lab Project</p>
# 
# <strong>Libraries Used</strong>
# <ul><li>pandas</li>
#     <li>matplotlib</li>
#     <li>seaborn</li>
#     <li>numpy</li>
#     <li>datetime</li>
#     <li>folium</li>
#     <li>bokeh</li>
# </ul>
# 
# <strong>Team Members</strong>
# <ul><li>Ramshankar</li>
#     <li>Srikanth</li>
#     <li>Manishankar</li>
#     <li>Jaswanth</li>
# </ul>
# 
# <p><strong>Size of Dataset : </strong>65 MB</p>

# <h3>We need to do the following in our whole project</h3>
# 
# <strong>Common Crimes in Chicago</strong> - done
# <ul><li>A graph depicting the most occurring offences in Chicago in 2018.</li>
#     <li>A comparison of the frequency of the top 5 most commonly committed offences</li>
#     <li>How likely are you to get arrested if you commit a crime?</li>
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

# In[3]:


import bokeh


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime


# <h2>Data Acquisition</h2>

# In[2]:


# Loading the csv dataset into the df variable
df = pd.read_csv('crimes_2018.csv')


# <h3>Scanning the data</h3>

# In[4]:


# The number of rows and columns in the dataset; returns a tuple
df.shape


# In[5]:


# Summary Statistics of the data
df.describe()


# In[6]:


df.info()


# In[7]:


sns.set_style('darkgrid')


# In[8]:


# First 5 rows of our dataset
df.head()


# <h4>What are our Features?</h4>

# In[9]:


# The names of the features
print("The names of the features :\n", list(df.columns))


# <h3>Number of Distinct Crimes in the city of Chicago in 2018</h3>

# In[10]:


# Number of distinct crimes in the city in 2018
crimes = df['Primary Type'].unique()
print("The Number of distinct crimes in Chicago in the year 2018:", len(crimes))
print()
print("The Distinct Crimes are :\n", crimes)


# <h2>Data Cleaning</h2>

# In[11]:


# What are the total missing values in the dataset ?
print("Number of Missing Values in the whole dataset : ", df.isna().sum().sum())


# In[12]:


# Let's count number of null entries per feature
missing_values = list(df.isna().sum())
# missing values is a list of the number of missing values in each column

cols = list(df.columns)
print(cols)
for i in range(len(cols)):
    if (missing_values[i] == 0):
        cols[i]="Others"
print(cols)
d = dict(zip(cols, missing_values))
print


# In[13]:


# Plotting the missing values in the dataset
x = list(d.keys())
y = list(d.values())
sns.barplot(x=x, y=y, palette="GnBu_d")
plt.xticks(rotation=90)
plt.title("Missing Values in the Dataset", fontdict = {'fontsize': 20})
plt.ylabel("Count of missing values", fontdict={'fontsize': 15})
plt.show()


# <h4>A Bit about the Missing values</h4>
# 
# <strong>Why is data missing in the first place ?</strong>
# <p>Inspecting the features, we see that all the features that have a large count of missing values are features that relate to the geographical location of the crime scene. This is <strong>No Surprise</strong> as the Chicago Crime Dataset is based on first hand accounts of people involved in or around the crime. It is not necessary that such first hand reports need to contain the specific locations of the crime.<br>Therefore, these missing values can be perfectly accounted for</p>
# 
# 
# <p>We have 11,238 missing values in the whole dataset that are present in Location Description, Ward, Community Area, X Co-ordinate, Y Co-ordinate, Latitude, Longitude and Location.<br>Since, these features are not direct numeric values, we can't use summary statistical functions to fill in the missing values.<br><strong>Hence, we shall be removing these values from the dataset</strong></p>

# In[14]:


# The simplest cleaning technique here would be to drop all the rows with atleast one missing value
df = df.dropna()
df.info()


# In[15]:


# How much of the data has been retained after this removal ?
print(round(262960 / 265698 * 100,2), "percentage of the data has been retained.")


# <p>Dropping the rows will usually result in <strong>clean datasets and produce well-behaved</strong> data. But often, it removes a lot of information that reduces result accuracy.<br>However, in our case, since <strong>98.97% of the data</strong> is retained and since there is practically no other way to work around the type of missing values we have, we shall go ahead with this slightly diminished dataset</p>

# In[16]:


# Continuous Variables
cont = df._get_numeric_data().columns
print("The continuous variables are: ",list(cont))


# In[17]:


# Categorical Variables
print("The categorical variables are: ",list(set(df.columns) - set(cont)))


# In[18]:


# Let's inspect the data once more
df.head()


# In[19]:


df.columns


# In[20]:


df.head()


# <h4>Check for available plot styles</h4>

# In[21]:


# use plt.style.available to view all possible styles


# <h3>Top Crimes in Chicago in the year 2018</h3>

# In[22]:


# Set the style of the plot first
plt.style.use('ggplot')
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


# <h3>Top 5 Crimes in Chicago in 2018</h3>
# <ul><li>Theft</li>
#     <li>Battery</li>
#     <li>Criminal Damage</li>
#     <li>Assault</li>
#     <li>Deceptive Practice</li>
# </ul>

# In[23]:


df.info()


# In[24]:


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


# In[25]:


month = s1[:2]
date = s1[3:5]
year = s1[6:10]

final_date = datetime(int(year), int(month), int(date), int(hr), int(mins), int(sec))
print(final_date)


# <h4>Convert the date-time column to a known format</h4>

# In[26]:


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


# In[27]:


# Using apply() of pandas to apply time_convert on every row of the Date column
df['Date'] = df['Date'].apply(time_convert)


# In[28]:


# Testing the status of AM or PM
df['Date'][0].strftime("%p")


# In[29]:


df['Date'].head(20)


# <h4>Create a new column "Month"</h4>

# In[30]:


def month(x):
    return x.strftime("%B")
df['Month'] = df['Date'].apply(month)


# In[31]:


df.head()[:5]


# <h4>The number of crimes in Chicago in 2018 as per month</h4>

# In[32]:


# Set plot style
plt.style.use('fivethirtyeight')
sns.set_context('notebook')

# Code to plot
sns.countplot(y='Month', data=df, palette=["#bb0e14"], order=['January', 'February', 'March', 'April', 'May', 'June', "July", 'August', 'September', 'October', 'November', 'December'], alpha=0.25)

# Aesthetic appeal of the plot 
plt.title("CRIMES PER MONTH\n", fontdict={'fontsize': 20, 'color': 'teal'}, weight="bold")
plt.ylabel("Month\n", fontdict={'fontsize': 15}, weight="bold")
plt.xlabel("\nNumber of Crimes", fontdict={'fontsize': 15}, weight="bold")
plt.show()


# <h3>Crime Patterns across months</h3>
# <p>The months of May, June, July and August have seen the most spike in crime rates in the city.</p>

# In[33]:


# Crimes per month 
theft_dict ={} # dictionary
battery_dict = {}
crim_dam = {}
assault = {}
dec_prac = {}

months = df["Month"].unique()
for month in months :
    theft_dict[month]=0
    battery_dict[month]=0
    crim_dam[month]=0
    assault[month]=0
    dec_prac[month]=0

for elem in df[df["Primary Type"]=="THEFT"]["Month"]:
    if elem in theft_dict.keys():
        theft_dict[elem] += 1

for elem in df[df["Primary Type"]=="BATTERY"]["Month"]:
    if elem in battery_dict.keys():
        battery_dict[elem] += 1
        
for elem in df[df["Primary Type"]=="CRIMINAL DAMAGE"]["Month"]:
    if elem in crim_dam.keys():
        crim_dam[elem] += 1
        
for elem in df[df["Primary Type"]=="ASSAULT"]["Month"]:
    if elem in assault.keys():
        assault[elem] += 1
        
for elem in df[df["Primary Type"]=="DECEPTIVE PRACTICE"]["Month"]:
    if elem in dec_prac.keys():
        dec_prac[elem] += 1


# In[34]:


# Plotting the graphs

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10,10))
x,y = zip(*theft_dict.items())
sns.lineplot(x,y, color="black", ax=ax)

x,y = zip(*battery_dict.items())
sns.lineplot(x,y, color="red", label='Battery')

x,y = zip(*crim_dam.items())
sns.lineplot(x,y, color="blue", label='Criminal Damage')

x,y = zip(*assault.items())
sns.lineplot(x,y, color="orange", label='Assault')

x,y = zip(*dec_prac.items())
sns.lineplot(x,y, color="green", label='Deceptive Practice')

ax.legend(['Theft', 'Battery', 'Criminal Damage', 'Assault', 'Deceptive Practice'], facecolor='w')

for tick in ax.get_xticklabels():
    tick.set_rotation(90)

ax.set(title="Frequency of Top 5 Crimes", xlabel="Month", ylabel="Number of Crimes")
    
plt.show()


# In[35]:


df.head()


# <h3>Escaping an Arrest</h3>

# In[36]:


# How are arrests spread out across the months
plt.style.use('bmh')

fig, ax = plt.subplots(figsize=(10, 5))
ax = sns.countplot(x="Month",
                   hue='Arrest',
                   data=df[['Month','Arrest']],
                   palette=['Red', 'Green'])
months = ['January','February','March','April','May','June','July',             'August','September','October','November','December']    

ax.set(title='Arrests Made per Month in 2018', xlabel='Month', ylabel='Number of Arrests', xticklabels=months)
plt.title('Arrests Made per Month in 2018', fontdict={'fontsize': 20, 'color': 'black'}, weight="bold")
plt.show()


# In[37]:


n = df['Arrest'].value_counts()
print(n)


# <p>That's a stoking <strong>80%</strong> chance for evading an arrest!</p>

# <h4>Create a new column "Hour" [24 hour format]</h4>

# In[38]:


def hour(x):
    return x.strftime("%H")
df['Hour_Day'] = df['Date'].apply(hour)


# In[39]:


df.head()


# <h4>Distribution of crimes in Chicago in 2018 as per hour</h4>

# In[40]:


# Set plot style
plt.style.use('seaborn-dark')
sns.set_context('paper')

# Write code to plot
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(x='Hour_Day', data=df, palette="viridis")

# Aesthetic appeal
plt.title("UNSAFEST HOURS\n", fontdict={'fontsize': 20, 'color': '#bb0e14'}, weight="bold")
plt.xlabel("\nHour in the Day", fontdict={'fontsize': 15}, weight='bold')
plt.ylabel("Number of Crimes\n", fontdict={'fontsize': 15}, weight="bold")

# Add Text to the plot
plt.text(2.3, 7000, 'Lowest Crime Rate', fontdict={'fontsize': 14, 'color':"green" }, weight='bold')

plt.show()


# In[41]:


df.head()


# In[52]:


# Which crimes are more prone to happen in the cover of darkness ?
dark_hours = ['19', '20', '21', '22', '23', '00', '01', '02', '03','04','05','06','07']


# In[ ]:





# In[ ]:





# In[37]:


df.head()


# In[38]:


df["Block"].value_counts()[:10]


# ### Crime Locations

# In[39]:


df.info()


# In[40]:


df.columns


# In[41]:


df.head()


# In[42]:


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


# In[43]:


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


# In[44]:


df.head()


# In[45]:


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

# In[ ]:




