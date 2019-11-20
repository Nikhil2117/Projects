#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[22]:


df = pd.read_csv("movies.csv")


# In[23]:


features = ['keywords','cast','genres','director']


# In[24]:


def combine_features(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']


# In[25]:


for feature in features:
    df[feature] = df[feature].fillna('')


# In[26]:


df["combined_features"] = df.apply(combine_features,axis=1)
#applying combined_features() method over each rows of dataframe and storing the combined string in "combined_features" column


# In[27]:


df.iloc[0].combined_features


# In[28]:


#creating new CountVectorizer() object
cv = CountVectorizer()
#feeding combined strings(movie contents) to CountVectorizer() object
count_matrix = cv.fit_transform(df["combined_features"])


# In[29]:


cosine_sim = cosine_similarity(count_matrix)


# In[30]:


def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]
def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]


# In[31]:


movie_user_likes = input("Enter Movie Name")
movie_index = get_index_from_title(movie_user_likes)
similar_movies = list(enumerate(cosine_sim[movie_index]))
#accessing the row corresponding to given movie to find all the similarity scores for that movie and then enumerating over it


# In[32]:


sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]


# In[34]:


i=0
print("Top 5 similar movies to "+movie_user_likes+" are:\n")
for element in sorted_similar_movies:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>10:
        break

