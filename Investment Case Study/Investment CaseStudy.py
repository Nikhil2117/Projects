# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(nikhil2117)s
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns',100)

company_url = 'https://raw.githubusercontent.com/akjadon/Finalprojects_DS/master/Bank%20Loan%20Default%20-%20Casestudy/companies.txt'

rounds2_url = 'https://raw.githubusercontent.com/akjadon/Finalprojects_DS/master/Bank%20Loan%20Default%20-%20Casestudy/rounds2.csv'

company = pd.read_csv(company_url,sep='\t', encoding='ISO-8859-1')
rounds2 = pd.read_csv(rounds2_url, sep =',',encoding='ISO-8859-1')

company.columns
rounds2.columns

company.head()
rounds2.head()


company.shape
rounds2.shape

company.isna().sum()
rounds2.isna().sum()


#%%% DATA CLEANING AND VARIABLE REDUCTION


#Unique Companies in rounds2
rounds2['company_permalink'] = rounds2['company_permalink'].apply(lambda x:x.lower())
rounds2['company_permalink'].unique().size

#Unique Companies in company
company['permalink'] = company['permalink'].apply(lambda x:x.lower())
company['permalink'].unique().size


#PrimaryKey in Company
company.head()
y = company.shape[0]-(company['permalink'].unique().size)
y
#Therefore the permalink could be the Primary key
print('Therefore the permalink could be the Primary key')

#Get list of companies which are there in rounds2 but not in company dataframe
rounds2[~rounds2['company_permalink'].isin(company['permalink'])]
#number of companies
size = rounds2[~rounds2['company_permalink'].isin(company['permalink'])].size

if size == 0:
    print('No')
else:
    print('Yes')
    

#to join the companies we have to perform merging in both the dataframes on 'company_permalink' :
#for performing the join the column must be same
rounds2.rename(columns={'company_permalink':'permalink'}, inplace=True)
rounds2.columns
master_dataframe = company.merge(rounds2, on = 'permalink', how = 'inner')
master_dataframe.shape
master_dataframe.info()
master_dataframe.head()
master_dataframe.columns


#Now removing useless columns 
master_dataframe.drop(['homepage_url','founded_at','funding_round_code','state_code','region','city'],inplace = True,axis = 1)

#Now Treating the Null values in DataFrame
master_dataframe.permalink.isnull().sum()

master_dataframe.name.isnull().sum()
#since the value cannot be imputed in this therefore droping this row
index = master_dataframe[master_dataframe.name.isnull()].index
master_dataframe.drop(index,inplace=True)

#removing rows with null values greater than 3
master_frame= master_dataframe[master_dataframe.isnull().sum(axis=1)<=3]

#removing the rows having null values in raised_amount_usd
master_frame= master_frame[~master_frame.raised_amount_usd.isnull()]
master_frame.raised_amount_usd.isnull().sum()
master_frame.shape
master_frame.columns

#treating the rows having null values in category_list
master_frame.category_list.isnull().sum()
master_frame= master_frame[~master_frame.category_list.isnull()]

master_frame.country_code.isnull().sum()
master_frame= master_frame[~master_frame.country_code.isnull()]

master_frame.isnull().sum()
master_frame.info()

master_frame.head()
#retained data
retained_data= round(100*len(master_frame.index)/114942,2)

retained_data



#%% 
master_frame.columns
#best_funding_type with mean btw 5-15 million USD

master_frame.groupby('funding_round_type')['raised_amount_usd'].count()
#venture is observed to be the best Funding Type

venture_data = master_frame[master_frame['funding_round_type']=='venture']

#top 9 countries within the venture_data
countries = venture_data.groupby('country_code').sum()
countries
top9 = countries.sort_values(by='raised_amount_usd', ascending = False).head(9)
top9

#%%
#Identifying top3 countries speaking English with top investment
venture_data.head()
top3 = venture_data.loc[(venture_data['country_code']=='IND') | (venture_data['country_code']=='USA')| (venture_data['country_code']=='GBR')]
top3.head(10)

top3.shape

top3['category_list']
top3.columns
#Now splitting the category list based on delimiter and added into new column in top3 dataframe
category_data = top3['category_list'].apply(lambda x: x.split('|')[0])
category_data
top3['primarysector'] = category_data
top3.head(10)
top3.primarysector
top3.columns

mapping_url = 'https://raw.githubusercontent.com/akjadon/Finalprojects_DS/master/Bank%20Loan%20Default%20-%20Casestudy/mapping.csv'

mapping_data = pd.read_csv(mapping_url, sep = ',')
mapping_data.info()
mapping_data.head()
mapping_data.columns

#converting wide dataframe into long dataframe using melt function
variables = mapping_data.columns[1:]
values = mapping_data.columns[:1]
values
variables
mapping_long= pd.melt(mapping_data, id_vars=list(values), value_vars=list(variables),var_name='main_sector', value_name='Count')
mapping_long.columns
mapping_long.rename(columns = {'category_list':'primarysector'},inplace = True)

mapping_long = mapping_long[mapping_long.Count == 1]
mapping_long

dataframe = pd.merge(top3, mapping_long,how = 'inner', on='primarysector')
dataframe

#Now getting the data of US based funding types with raised amount between 5 to 15 million USD

us_data = dataframe.loc[(dataframe.raised_amount_usd >= 5000000) & (dataframe.raised_amount_usd <= 15000000) & (dataframe.country_code=='USA')]

us_data.head()

#Now getting the data of The Great Britain based funding types with raised amount between 5 to 15 million USD

gbr_data = dataframe.loc[(dataframe.raised_amount_usd >= 5000000) & (dataframe.raised_amount_usd <= 15000000) & (dataframe.country_code=='GBR')]

gbr_data.head()

#Now getting the data of India based funding types with raised amount between 5 to 15 million USD

ind_data = dataframe.loc[(dataframe.raised_amount_usd >= 5000000) & (dataframe.raised_amount_usd <= 15000000) & (dataframe.country_code=='IND')]

ind_data.head()
ind_data.columns


#%%performing analysis on USA data

us_data.head()
#total investment that took place 
us_data.groupby('main_sector')['raised_amount_usd'].describe().sum().sum()
#total number of investment
us_data.groupby('main_sector')['Count'].describe().sum().sum()
#Top sectors for investments
us_data.groupby('main_sector')['raised_amount_usd'].describe()
#it is clear from above that others sector is 1st top investing sector,Cleantech / Semiconductors is 2nd 
#Company receiving highest investment for top sector ('Others')
us_data[us_data['main_sector']=='Others'].groupby('name')['raised_amount_usd'].sum().sort_values(ascending=False).head()

#Company receiving highest investment for 2nd Top Sector Cleantech / Semiconductors
us_data[us_data['main_sector']=='Cleantech / Semiconductors'].groupby('name')['raised_amount_usd'].sum().sort_values(ascending=False).head()

#%%
#%%performing analysis on The great britain data

gbr_data.head()
#total investment that took place 
gbr_data.groupby('main_sector')['raised_amount_usd'].describe().sum().sum()
#total number of investment
gbr_data.groupby('main_sector')['Count'].describe().sum().sum()
#Top sectors for investments
gbr_data.groupby('main_sector')['raised_amount_usd'].describe()
#it is clear from above that others sector is 1st top investing sector,Cleantech / Semiconductors is 2nd 
#Company receiving highest investment for top sector ('Others')
gbr_data[gbr_data['main_sector']=='Others'].groupby('name')['raised_amount_usd'].sum().sort_values(ascending=False).head()

#Company receiving highest investment for 2nd Top Sector Cleantech / Semiconductors
gbr_data[gbr_data['main_sector']=='Cleantech / Semiconductors'].groupby('name')['raised_amount_usd'].sum().sort_values(ascending=False).head()


#%%performing analysis on India data

ind_data.head()
#total investment that took place 
ind_data.groupby('main_sector')['raised_amount_usd'].describe().sum().sum()
#total number of investment
ind_data.groupby('main_sector')['Count'].describe().sum().sum()
#Top sectors for investments
ind_data.groupby('main_sector')['raised_amount_usd'].describe()
#it is clear from above that others sector is 1st top investing sector  News, Search and Messaging  is 2nd 
#Company receiving highest investment for top sector ('Others')
ind_data[ind_data['main_sector']=='Others'].groupby('name')['raised_amount_usd'].sum().sort_values(ascending=False).head()

#Company receiving highest investment for 2nd Top Sector Cleantech / Semiconductors
ind_data[ind_data['main_sector']=='News, Search and Messaging'].groupby('name')['raised_amount_usd'].sum().sort_values(ascending=False).head()

#%%
#Box and count plot to show average investments and number of investments
filtered_df = master_frame.loc[(master_frame['funding_round_type']=='venture') | (master_frame['funding_round_type']=='seed') | (master_frame['funding_round_type']=='private_equity')]

filtered_df.columns
# subplot 1: Mean
plt.subplots(figsize=(20,10))
plt.subplot(1, 3, 1)
axis_bar = sns.barplot(x='funding_round_type', y='raised_amount_usd', data=filtered_df)
axis_bar.set(xlabel='Investment Type', ylabel='Average Investment Amount (USD) in Millions')
plt.title('Average investment amount across investment types')

# subplot 2: No of investments
plt.subplots(figsize=(20,10))
plt.subplot(1, 3, 3)
axis_count = sns.countplot(x='funding_round_type' , data=filtered_df)
axis_count.set(xlabel='Investment Type', ylabel='No of Investments')
plt.title('No of investments')
plt.show()

#Get the data frame consisting of top 9 countries investment
top9_df = pd.DataFrame({'country_code':top9.index, 'raised_amount_usd':top9.raised_amount_usd})
top9_df

#Bar plot for total no of investments by country code
plt.subplots(figsize=(10,8))
axis = sns.barplot(x='country_code', y='raised_amount_usd', data=top9_df)
plt.yscale('log')
axis.set(xlabel='Country Code', ylabel='Investment Amount Millions of USD')
plt.title('Amount of Investments by country')
plt.show()

#Get all the sectors with investment range between 5 to 15 million
sector_df = dataframe.loc[(dataframe.raised_amount_usd >= 5000000) & (dataframe.raised_amount_usd <= 15000000)]

#Get the investments in the main sectors by the country code
top3_df = sector_df['main_sector'].groupby(sector_df['country_code']).value_counts()
top3_df
#Get the top 3 sectors in top 3 countries in a dataframe
top3_filtered = pd.DataFrame({'count' : top3_df.groupby( 'country_code').head(3)}).reset_index()
top3_filtered
plt.subplots(figsize=(10,8))
#Plot to show the no of investments in top 3 countries in top 3 sectors
axis = sns.barplot(x='country_code', y='count', hue='main_sector', data=top3_filtered)
plt.yscale('log')
axis.set(xlabel='Country Code', ylabel='No of investments')
plt.title('No of investments in top 3 countries in top 3 sectors')
plt.show()
