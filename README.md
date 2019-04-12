# Chicago-Crime-Data-Analysis


Using Data Science to explore, analyze and visualize the Chicago Crime Dataset
Keywords : Chicago Crime Dataset, Data Science, Exploratory Data Analysis (EDA), Data Visualization

Crime has been a prevalent anti-social trait in the human society since time immemorial and it continues to be so even today. Surging crime rates and faulty policies to prevent prevailing crimes have been major issues which the world is fighting against, even in the year of 2019.
According to several noted writers, researchers and journalists, crime follows a pattern. Especially if crimes are organized and conducted by well defined, well organized gangs. Understanding these crime patterns and criminal behaviour would largely strengthen the police force in dealing with wrongdoings. A region prone to crime during a particular hour can be under closer watch with a larger force of police deployed in the area to maintain peace. 
Through our project, we aim to analyse patterns in crime and unearth insights into several questions that can aid police to understand crime patterns and also will help civilians understand the truth as to how safe they are in reality. We shall be using data science methodologies in performing our analysis and shall follow a method very close to the following graphic :
 
Fig. 1) The Data Science Workflow
The Dataset Used :
The Chicago Crime Dataset provided by the Chicago Data Portal will be used for our project. The Chicago Police Department has registered numerous criminal cases daily since 2001 and has made this data available publicly on their website : https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2 
Data is extracted from the Chicago Police Department's CLEAR (Citizen Law Enforcement Analysis and Reporting) system. The original complete dataset contains more than 65,000 records/rows of data and 22 columns/features. The dataset has crimes from 2001 to 2019 (minus the most recent 7 days). The dataset of gargantuan proportions has a size of 1.5 GB.
However, since the team found it difficult to deal with the large amount of data on our systems, we have considered only the “Crimes of 2018”. The size of this dataset is 65 MB. But the team shall be working on this data keeping in mind the reason that we will have to extend our working code to the whole complete dataset of 1.5 GB if we choose to extend this project later on. 
Focus of the project :
The focus of our project is to perform an in-depth analysis of the major types of crimes that occurred in the city, observe the trend over the months, and determine how various attributes, such as seasons, time of the day like daytime or night affect the occurence of crimes. The analysis is also done to determine which area has the highest crimes on the basis of the crime categories, etc. Our project will be aimed at providing effective visualizations to unearth patterns and answers to several questions we had in mind before starting our implementation. 

Main Categories of Questions :
The following are the main categories of questions that we have worked on in our project :

1.	Common Crimes in Chicago
2.	Crime vs Time
3.	Crime vs Locations
4.	Gang Related Offences
Area of Topic Chosen - Data Science
Our project is largely centered around the thrust area of Data Science. Data science is a multidisciplinary blend of data inference, algorithm development, and technology in order to solve analytically complex problems. In simpler words, data science is all about collecting data, making it usable, applying mathematical and statistical tools onto it and making inferences of the data; finally coupling these inferences and results with domain knowledge/business acumen and helping transform businesses and the lives of people for the better. 
Data Science, as is explained through Fig. 2) is a broad encapsulation of a wide range of skills. 

 
Fig. 1) What is Data Science ?

The Scope of Data Science in this project :
In order to understand the pattern of crimes and analyze the rate of crime in Chicago, it is important that we have in possession to the dataset of crimes in Chicago. With the involvement of data, Data Science becomes an inevitable concept that intertwines into the project. 
For our project, we have used a variant of the OSEMN Framework (Fig. 3). 
 
Fig. 3) The OSEMN Data Science Framework

Obtaining Data: The dataset was obtained from the Chicago Police Department's CLEAR (Citizen Law Enforcement Analysis and Reporting) system. In our project, we have used crimes_2018.csv (The crime dataset for 2018) [Size : 65 MB]
The data was a .csv file. We used read_csv() of the pandas module for loading our dataset into a dataframe.
Scrubbing Data: The dataset obtained was not very messy, yet there were issues with the data that we had to resolve before we could work on it. There were a few missing values that were removed by dropping rows with missing values (1.03% of the whole data was removed in this process). Also, the Date column in the original dataset was modified using python’s datetime module. 
Also, feature engineering was performed to create columns such as “Month”, “Hour” etc. to simplify the analysing process.
Exploring Data: This was the major part of our project. Exploring the data and understanding it and using our understanding to find answers to several questions we had formulated was the most important part of our project. For this purpose, we used in-built modules, made our own user defined functions, created effective plots and visualizations, mathematical understanding in concepts of probability were also employed in areas.
Modelling Data: Creating forecasting models was not a major thrust area in our work. However, the team is confident of being able to create forecasting modules in the same project in a future continuation of the same.
Interpret Data: With the help of our plots and findings, we were able to apply our intuition onto the data and come up with plausible explanations to our major questions regarding the data. We have made attempts to effectively communicate our results of the case study on the chicago crime dataset for the year 2018.
The Specialized Area the project focuses on :
Although the broad idea of our project uses the data science methodologies, our specialized view is centered on “Data Visualization Methods” for effective communication with data. The main aim of our case study is to help stakeholders (police, civilians in our case) to be able to make sense of raw data. 









System Model/Design
The scope of our project does not deal with creating any kind of software or package. Hence, there is no system as such to explain through a use case model. However, we have created an infographic below that will help understand the design of our project and the way it has been structured.
 
Fig. 4) The System Design for the project
Implementation Details

About Modules Used :
●	Pandas-
         Pandas is an open source, BSD-licensed library providing high-performance,      easy-to-use data structures and data analysis tools for the Python programming language
●	Matplotlib-
        Matplotlib is a python library used to create 2D graphs and plots by using python scripts. It has a module named pyplot which makes things easy for plotting by providing feature to control line styles, font properties, formatting axes etc.
●	Seaborn-
           Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
●	Numpy-
           NumPy is a Python package which stands for 'Numerical Python'. It is a              library consisting of multidimensional array objects and a collection of routines for processing of array.
●	Datetime-
            The datetime module supplies classes for manipulating dates and times in both simple and complex ways. While date and time arithmetic is supported, the focus of the implementation is on efficient attribute extraction for output formatting and manipulation.

●	Folium-
           Folium makes it easy to visualize data that’s been manipulated in Python on an interactive leaflet map. It enables both the binding of data to a map for choropleth visualizations as well as passing rich vector/raster/HTML visualizations as markers on the map.
●	Bokeh-
            Bokeh is an interactive visualization library that targets modern web browsers for presentation. Its goal is to provide elegant, concise construction of versatile graphics, and to extend this capability with high-performance interactivity over very large or streaming datasets

Sample Code :
# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import bokeh 

# Loading the dataset
df = pd.read_csv(“crimes_2018.csv”)

# Cleaning the dataset
missing_values = list(df.isna().sum())
cols = list(df.columns)
print(cols)
for i in range(len(cols)):
    if (missing_values[i] == 0):
        cols[i]="Others"
df = df.dropna()
df.info()

# Plotting the top crimes in Chicago for 2018
plt.style.use('ggplot')
sns.set_context('notebook')
top_5_crimes=df['PrimaryType'].value_counts().sort_values(ascending=False).head()
temp = df.groupby('Primary Type', as_index=False).agg({"ID": "count"})
temp = temp.sort_values(by=['ID'], ascending=False).head()
sns.barplot(x='Primary Type', y='ID', data=temp, palette="Blues_d")
plt.title("Top 5 Crimes in Chicago\n", fontdict = {'fontsize': 20, 'color': '#bb0e14'})
plt.ylabel("COUNT OF CRIMES", fontdict = {'fontsize': 15})
plt.xlabel("TYPE OF CRIME", fontdict = {'fontsize': 15})
plt.xticks(rotation=90)
plt.show()

# Converting the date-time column to a known format
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
            hr = '00’
    final_date = datetime(int(year), int(month), int(date), int(hr), int(mins), int(sec))
    return final_date

# Simulating the map of Chicago with the help of X and Y co-ordinates
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






Output
 
Fig. 5) Output 1
The head command in pandas helps us inspect our dataset as a dataframe (like a table)
 
Fig. 6) Output 2
In Fig. 6) , we inspect the different crimes that have been committed in Chicago.
 
Fig. 7) Output 3 - Missing Values in the dataset

 
Fig. 8) Output 4 - Most committed offences in Chicago in 2018

 
Fig. 9) Output 5 - Most committed offences in Chicago in 2018
The above visualization shows us that most crimes committed happen during the Summer!
 
Fig. 10) Output 6 - The Unsafest Hours in Chicago
The above visualization is a depiction of the most unsafest hours in Chicago. For this, we have feature engineered “Hours” column and mapped it to the count of crimes. According to the visualization, the lowest crime rate is during the hours of 01:00 a.m. - 05:00 a.m (As here, the graph is dropping). It picks up again after 05:00 a.m. 
This is a normal find as after all, “Criminals need some sleep too!”

 
Fig. 11) Output 7 - Arrests made per month in 2018 in Chicago

Clearly, a lot of arrests were not made in a lot of crimes throughout the year. 80% of the crimes saw No Arrests. But, since all the data collected was based on first hand accounts of crimes, the 80% figure should in reality be lesser and more arrests should have been made.




Conclusion
The use of data science to make decisions is key to improving lifestyles in the present world. Data driven decision making has proved to be highly profitable and life changing for several businesses. A very important example of such a scenario is the “Famous diaper-beer correlation” which helped Walmart place beer and diapers together for improving customer experience, which in turn helped Walmart’s business by causing an instant increase of 35% in the sales of both beer and diapers.
However, through our project, we are trying to throw light on the usually overlooked aspect of data science, “It’s ability to help understand our lives and improve human settlements in a better way”. Data is everywhere in our world, so finding data that will help solve a problem is not as hard as what it was a couple of decades ago. 
In our project, through the analysis of crime data, we were able to find out answers to several questions we had regarding the crimes in Chicago. Using a structured process as mentioned in Fig. 4) and backing it up with the OSEMN framework to conduct our research (Fig. 3)), we were able to achieve our results. A few of our finds included finding answers to questions like which were the most committed crimes in 2018, the unsafest hours, the unsafest locations, the worst places your car could have a breakdown during midnight hours, the most prevalent gang related activity in Chicago etc.
We also came across a startling fact that, no arrests were made in 80% of the crimes. But, it should also be noted that the data collected were first hand accounts, so it is more optimistic for us to state that 20% of the crimes committed saw arrests within the first reporting of the crime itself!
The project has a great deal of scope and with future work, predictive models can be built to help build software that would help the police identify areas of high alert during particular days and times. 
With the existence of such software in states within the US, it is important that such software is also extended to India with the rising crime rates in recent times. Availability of ready crime data is still a hurdle and if such data is made open to the public, it would be a great advantage to conduct research on the data and build models to aid in the policing process and Predictive Policing can be made into a reality in the country.
