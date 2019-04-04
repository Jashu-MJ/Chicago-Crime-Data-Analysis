
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime


# In[51]:


# Loading the csv dataset into the df variable
df = pd.read_csv('crimes_2018.csv')
c = df['Primary Type'].unique()
print(c,'\n')
print("assuming theft , homicide and narcotics as gang based criminal activities")

crimes = df['Primary Type'] 
#crimes.index=["HOMICIDE"]  
homicide = df[df['Primary Type']=='HOMICIDE']
theft = df[df['Primary Type']=='THEFT']
narcotics= df[df['Primary Type']=='NARCOTICS']

merge = homicide.append(theft)
merge = merge.append(narcotics)

crimes.head() 
#crimes.loc[['HOMICIDE'],: ]    
#cols = list(df.columns)
'''
for i in range(len(crimes)):
    if (i!="HOMICIDE"):
        df[i,:].drop()
#d = dict(zip(cols, missing_values))
'''

'''
for row in crimes:
    if(row!="HOMICIDE"&"THEFT"&"NARCOTICS"):
        
'''        
    

print('\n')

sns.lmplot('X Coordinate', 
           'Y Coordinate',
           data=theft,
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
