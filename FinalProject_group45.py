#!/usr/bin/env python
# coding: utf-8

# # COGS 108 - Final Project 

# # Overview

# The background of the  Spotify Tracks Database contains the songs related features including genre, artist, track, id, popularity and  so on. Each song in the Spotify database has a unique ID. We can apply the features such as genre and popularity to predict the popularity of a song based on its genre, and we can predict how is the popularity of a song being related to the singer. We can even combine multiple characteristics of a song to predict whether a song with certain crucial features will be most likely to be popular in the features. 
# The database we chose is newly updated in 2019 March, so very few online projects applied this database yet, but we can find some other Spotify database in previous years and applied the ideas from those database related projects.
# 
#  This data is extremely useful for artists or producers that are looking to create the next “hit” song.  By utilizing the given data, they can possibly predict what the next “hit” song will be by analyzing the data of song features and determining if there is a correlation.  We can apply the features such as genre and popularity to predict the popularity of a song based on its genre, and we can predict how is the popularity of a song being related to the artist. We can even combine multiple features of a song to predict whether a song with certain crucial features will be most likely to be popular in the features. We want to discover the specific song features that all top charting songs share, if there are any.  We will look at specifically chosen song features to see if there is a correlation between songs sharing similar features that are in the top charts.  If our hypothesis is correct, it will show that songs in music top charts share similar song features.  Thus, this will lead us to be able to predict the next “hit” song. 
# 
# 
# 

# # Names
# 
# -Yu Shen: set up, Data Cleaning, Target classification, Model Training, Data Analysis & Results
# 
# -Zhaokai Xu: set up, Data Cleaning, model training, Data Analysis & Results，Visualization 
# 
# -Iris Peng: 
# 
# -Anthony Martinez: 
# 
# -Jingwen Chen: 
# 

# # Group Members IDs
# 
# - A13496628
# - A14738474
# - A13696093
# - A14774741
# - A13378551

# # Research Question

# 
# Output target: song_popularity 
# 
# Q1. Do popular songs have similar attributes? If so, what are these attributes? Specifically, out of a chosen 13 attributes (song duration, tempo, key, etc.), are there similarities between them that determine a song’s popularity? 
# 
# Q2. Furthermore, after identifying these attributes, can we use them to predict which song will be most popular next? 	
# 
# 

# ## Background and Prior Work

# Music is a huge component of the history and culture of mankind.  More recently, music has evolved into a business.  Artists are constantly competing for the top spots on music charts.  Whether it is for top song or top album.  Being highly ranked in these charts is important for artists because it will lead to more endorsements, more recognition, and most importantly, more record and song sales.  Now, creating a song that makes it on the top charts doesn’t necessarily mean that it was composed by Mozart.  Most popular songs have a familiar sound.  Upbeat, fast, and loud are good indicators of what could possibly be a popular song.  Creating a popular song doesn’t necessarily mean that you will make it on the top charts, but it does give you a better chance at it.  Now, streaming is the future of music.Streaming services like Apple Music and Spotify are releasing very useful data on current music.  More specifically, Spotify releases data on the song features for each song.  There are 13 total song features that Spotify releases to the public including: tempo, time signature, loudness, danceability, and more. 
# 
# In other Spotify related projects such as, https://www.kaggle.com/nadintamer/top-tracks-of-2017  
# It raises interesting questions to explore the dataset such as 
# Looking for patterns in the audio features of the songs. Why do people stream these songs the most?
# How can we predict one audio feature based on the others features 
# Explore which features correlate the most
# https://www.kaggle.com/edumucelli/spotifys-worldwide-daily-song-ranking
# Can you predict what is the rank position or the number of streams a song will have in the future?
# What are the signs of a song that gets into the top rank to stay?
# Do continents share same top ranking artists or songs?
# Are people listening to the very same top ranking songs on countries far away from each other?
# Other projects builders mostly applied machine learning regression models and classification models to make predictions, and the most popular library they referred to is Sklearn. They realized that the data preprocessing is crucial to the success of the projects, and feeding balanced data into the model they applied greatly contributes to improve the accurate prediction.These are the things we can be careful about while working on our project.
# Since the dataset we are using contains different features compared to the other Spotify dataset, we suggest to try to take the best advantage of our unique features and raise some interesting exploration.
# References (include links):
# - 1) https://www.kaggle.com/nadintamer/top-tracks-of-2017  
# - 2) https://www.kaggle.com/edumucelli/spotifys-worldwide-daily-song-ranking

# # Hypothesis
# 

# Among the 14 features we have, 'song_duration_ms', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature', 'audio_valence', u'playlist, we want to analyze what feature would contribute the most to the popularity of a song. 
# 
# Intuitively, we will make the hypothesis that dancebility, livenesss, tempo and energy are key factors of a song_popularity

# # Dataset(s)

# - Dataset Name:19000 Spotify Songs 
# - Link to the dataset: https://www.kaggle.com/edalrami/19000-spotify-songs
# 
# - Number of observations:19000
# 
# This dataset uses Spotify API and contains data of 19000 songs. It has 15 features, including ：song_name, song_popularity,song_duration_ms, acousticness, danceability, energy, instrumentalness, key, liveness, loudness, audio_mode, speechiness, tempo, time_signature, audio_valence.  It has a relatively large amount of samples which is the main reason we chose this dataset. All these audio features come from Spotify API and the data set is updated 5 months ago.

# # Setup and data pre-analyzing

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

#TODO: NEED Explanation 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import patsy
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import ttest_ind, chisquare, normaltest, norm
import seaborn as sns;

from sklearn.model_selection import train_test_split  
from sklearn.utils import shuffle


# In[2]:


# 1.1 read data 

song_data_df = pd.read_csv("song_data.csv")

song_info_df = pd.read_csv("song_info.csv")


# In[3]:


# 1.2 preview data
song_data_df.head(5)


# In[4]:


song_info_df.head(5)


# In[5]:


song_data_df = song_data_df.set_index('song_name')
song_info_df = song_info_df.set_index('song_name')


# The reason we set the song_name as the index is: 
# 1. we do not want to treat it as a feature; 
# 2. using it as index allow us to combine two dataframe by index. 

# In[6]:


song_data_df.shape


# In[7]:


song_info_df.shape


# In[8]:


# 1.3 combine the two dataset into df
df = pd.concat([song_data_df, song_info_df], axis=1)
df.shape


# In[9]:


df.head(5)


# In[10]:


# 1.4 print distribution alongside all features
hist = df.hist(bins = 50,figsize = (15,20))


# In[11]:


song_popularity = df['song_popularity'].tolist()
# find stats of song__popularity: mean and standard deviation
mu, sd = norm.fit(song_popularity)

#draw the histogram of the target
plt.subplots(figsize=(15,10))
n, bins, patches = plt.hist(song_popularity, bins = 100,density = 1,alpha = 0.75)

#draw the distribution function curve 
y =stats.norm.pdf( bins, mu, sd)
curve = plt.plot(bins, y, 'r--', linewidth=2)
plt.show()


# #TODO Analyze the distribution here. 
# 

# # Data Cleaning

# Describe your data cleaning steps here.

# ### 2.1 drop potential useless features: 

# In[12]:


df = df.drop(['album_names'], axis=1)
df = df.drop(['artist_name'], axis=1)


# We drop these two features because they all have unique values, which cannot be treated as a meaning feature. Therefore we dropped them. 

# ### 2.2 one hot encoding the playlist

# In[13]:


newdf = pd.get_dummies(df,prefix=['playlist'])


# In[14]:


newdf.head(5)


# After one-hot encoding, we have 313 features, 300 of which are one-hot encoded of different playlists, since we have 300 different playlists in the dataset. 

# ### 2.3 analyze data relevance 

# In[15]:


len(df['playlist'].unique())


# In[16]:


# analyze correlation 
# in range [-1,1], -1 means negtively correlated, 1 means positively correlated,0 means no relation

# analyze the correlation corresponding to popularity 


sns.set(style="white")

# Generate a large random dataset
# rs = np.random.RandomState(33)
d = pd.DataFrame(data=df,columns=list(df.columns))

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# #TODO: Analysis here

# In[17]:


# This class maps each variable in a dataset onto a column and row in a grid of multiple axes. 
# Different axes-level plotting functions can be used to draw bivariate plots in the upper and lower triangles, 
# and the the marginal distribution of each variable can be shown on the diagonal.
g = sns.PairGrid(df)
g = g.map(plt.scatter)


# #TODO: Analysis here

# In[18]:


# print the corr matrix for later filtering use
corr = df.corr()
corr.style.background_gradient()


# Here, we run the correlation function, to observe the correlation between two features. It returns a value from -1 to 1, representing how the two variables are related. 
# 
# Since the target is song_popularity, in the following step, we will drop the features that are not too relevant to song_popularity. We set the threshhold as 0.05. This means, if the correlation of a feature between song_popularity, is less than 0.05, we will drop it in the dataframe. 

# In[19]:


# drop the features whose correlation is less than 0.05

dropIndex = corr[abs(corr['song_popularity'])<0.05].index
print(dropIndex)

for name in dropIndex:
    df = df.drop([name], axis=1)
    newdf = newdf.drop([name], axis=1)

newdf.head(5)


# In[20]:


newdf.shape


# In[21]:


### Data Wraggling


# In[22]:


# separate the training and testing
X = df.drop(['song_popularity','playlist'], axis=1)
Y = df['song_popularity']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)


# In[23]:


print("X_train:",X_train.shape, "y_train",y_train.shape)
print("X_test:",X_test.shape, "y_test",y_test.shape)


# In[24]:


X.head()


# In[25]:


Y.head()


# # Train the model

# Before training the model, we want to shuffle it to make sure we are training using a random dataset. 

# In[26]:


# shuffle the data 
X = shuffle(X)


# In[27]:


X.head()


# In[28]:


# visualization
fig = plt.figure()
ax = plt.axes()
##TODO


# ## 3.1 train a parametric model 
# 
# #### Benefits of Parametric Machine Learning Algorithms:
# 
# Simpler: These methods are easier to understand and interpret results.
# Speed: Parametric models are very fast to learn from data.
# Less Data: They do not require as much training data and can work well even if the fit to the data is not perfect.
# Limitations of Parametric Machine Learning Algorithms:
# 
# ##### Constrained: By choosing a functional form these methods are highly constrained to the specified form.
# Limited Complexity: The methods are more suited to simpler problems.
# Poor Fit: In practice the methods are unlikely to match the underlying mapping function.
# 

# ### Multi-Variable Linear Regression
# 
# statsmodels.regression.linear_model.OLS

# In[29]:


# set song_popularity to be predicted value y
# set (acousticness	 danceability	instrumentalness	loudness	audio_valence	playlist) to be X

# Note the difference in argument order
model = sm.OLS(np.asarray(y_train), np.asarray(X_train)).fit()
y_pred = model.predict(np.asarray(X_test)) # make the predictions by the model


# In[30]:


model.summary()


# In[31]:


#print('params', model.params)
print('tvalues',model.tvalues)

print("The model is:","\n song_popularity = ", model.params[0],"*acousticness","\n",
     model.params[1],"*danceability",model.params[2],"*instrumentalness","\n",
      model.params[3],"*loudness",model.params[4],"*audio_valence")


# In[32]:


#visualize the result 
df_reg = pd.DataFrame( y_test.values, columns = ['actual_song_popularity'], index=y_test.index)
df_reg['predict_song_popularity'] = y_pred
df_reg


# In[33]:


# to view the distribution difference
ax = df_reg.plot.hist(bins=12, alpha=0.5)


# In[34]:


# evaluate the test loss
def 
y_test.values


# ### Naive Bayes

# In[ ]:


#TODO
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_test)


# In[ ]:





# ## 3.2 train a non-parametic model: 
# 
# #### Benefits of Nonparametric Machine Learning Algorithms:
# 
#   Flexibility: Capable of fitting a large number of functional forms. 
# 
#   Power: No assumptions (or weak assumptions) about the underlying function. 
# 
#   Performance: Can result in higher performance models for prediction. 
# 
# #### Limitations of Nonparametric Machine Learning Algorithms:
# 
# More data: Require a lot more training data to estimate the mapping function.
# Slower: A lot slower to train as they often have far more parameters to train.
# Overfitting: More of a risk to overfit the training data and it is harder to explain why specific predictions are made.

# ### K-NN

# ###  Decesion tree 

# In[ ]:


df.head()


# In[ ]:





# ### SVM

# In[ ]:





# ### Classification
# We now want to convert the problem to a classification problem, by clustering the popularity to 10 different levels. 

# In[ ]:


#sort all Y data into 10 classification from 1-10


# # Data Analysis & Results

# Include cells that describe the steps in your data analysis.

# In[ ]:


# analyze 15 features, relevance


# In[ ]:





# In[ ]:


# model training, multi regression


# In[ ]:





# In[ ]:





# In[ ]:


## YOUR CODE HERE
## FEEL FREE TO ADD MULTIPLE CELLS PER SECTION


# # Ethics & Privacy

# We took our data from kaggle.com, which is a public database made specifically for data science work. We’ve only used and analyzed retrieved from the site, so permissions of use and privacy concerns have already been filtered by the site prior to use.
# 
# Because we took our data from the site, there are limitations to our analysis that stem from this. For instance, we can only analyze and use the song features that were listed in the song. However, by carefully choosing features that best characterize songs, we can still derive a close approximation. Another bias we should be mindful of in our analysis is that certain songs may be more similar to another because they are of the same genre. In this case, finding more salient features of the song such as tempo to be similar would be redundant. We can mitigate this issue, however, by also analyzing other features of the song that are not homogeneous across those of the same genre.
# 
# Additionally, although the usage of data may not have been an issue, we should still be considerate of the potential issues with distribution of the conclusions drawn from the data. For instance, that this information won’t result in an abusive inclusion of the features in hopes of gaining more popularity in a song. Although we, personally, may not use it for unethical marketing purposes, we should still try to ensure that such a thing may not happen.
# 

# # Conclusion & Discussion

# *Fill in your discussion information here*
# 
# Q1. Do popular songs have similar attributes?
# 
# Q2. If popular songs do have similar attributes, can we predict what song will be the next “hit”?
# 

# ## reference: 
# http://queirozf.com/entries/one-hot-encoding-a-feature-on-a-pandas-dataframe-an-example
# 
# https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9
# 
# https://machinelearningmastery.com/parametric-and-nonparametric-machine-learning-algorithms/
# 
# 

# In[ ]:




