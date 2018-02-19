
# coding: utf-8

# #### This notebook serves to explore titanic data more in depth to discover more relationships between columns and survival rate

# In[1]:


import sklearn
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
get_ipython().magic('matplotlib inline')
sns.set()


# In[2]:


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


# In[3]:


sns.factorplot(x='Survived', data=df_train, kind='count')


# ##### Now that we know the proportion of those who survived and those who didn't, we can make a baseline accuracy

# In[4]:


df_train.info()


# In[5]:


numOfNonSurvivors = df_train[df_train['Survived'] == 0].Survived.count()
totalPassengers = df_train.PassengerId.count()

# baseline accuraccy by perdicting all passengers die
baselineAccuraccy = numOfNonSurvivors/totalPassengers
print(baselineAccuraccy)


# ##### this means that the worse we can do by predicting that all passengers die is 61%
# ##### therefore, anything that we do to improve our model should be better than the baselineAccuraccy
# ##### we can start by looking and trying to find out certain relationships in the data wrt to survival of the passenger

# ## Age Vs Survival

# ##### now we see that, Age has over 100 missing values and we therefore must address this, but let's first take a look at the data and try to visualize it

# In[6]:


print(df_train.Age.max())
print(df_train.Age.min())
print(df_train.Age.mean())


# In[7]:


# we will extract the rows where Age is not Nan and look at its distribution 
df_Age_Not_Null = df_train[df_train.Age.isnull() == False]


# In[8]:


df_Age_Not_Null.info()


# In[9]:


# now we have all the rows where Age is not null and we will look at its distribution
sns.distplot(a=df_Age_Not_Null['Age'], kde=False)


# In[10]:


print('Min Age: ' , df_Age_Not_Null.Age.min())
print('Max Age: ' , df_Age_Not_Null.Age.max())
print('Median Age: ', df_Age_Not_Null.Age.median())
print('Mode Age: ',df_Age_Not_Null.Age.mode().loc[0])
print('=============')
df_Age_Not_Null[df_Age_Not_Null['Age'] == 24].info()


# #####  from the Age distribution, we can see that there are a lot of passengers in the 20 to 40 year-old range with 24 being the highest occuring age (30 people are aged 24)
# 
# ##### now let's see the survival rate of the people wrt to Age (we will visualize this using striplot!!)

# In[11]:


sns.stripplot(data=df_Age_Not_Null, x='Survived', y='Age', jitter=True, alpha=0.5)


# In[12]:


sns.swarmplot(x='Survived', y='Age', data=df_Age_Not_Null)


# <p>From both the plots above, we can see that the survival rate for those in between their 20s and 40s are much higher than those in other age groups although death rate is also highest in this age group simply because this age group is the largest in the passenger population</p>
# 
# ## Conclusion: survival rate for those in between their 20s and 40s are much higher than those in other age groups
# 
# # Sex vs Survival
# 
# <p>Now, we would also like to see if Sex has anything to do with Survival rate and if it does, is it also correlated with Age?</p>

# In[13]:


# first of all, let's view the male and female population in the passengers
sns.factorplot(x='Sex', data=df_train, kind='count')


# <p>So there's a lot more male than females in the passenger population (note that we are using df_train)</p>
# <p>Now, lets view the survival rate</p>

# In[14]:


sns.factorplot(x='Survived', data=df_train, hue='Sex', kind='count')


# <p>Looks like a lot more females survived compared to males. Maybe the males all ensured that their loved ones survive? Hard to say</p>
# <p>Let's see if Age in the females also affected their survival rate</p>

# In[15]:


# we will be using df_Age_Not_Null since we also want to see relationship between Age, Sex and Survival
sns.stripplot(x='Survived', y='Age', hue='Sex', data=df_Age_Not_Null, jitter=True, alpha=0.5)


# <p>From the plot above, we can see that most of the girls who survived are in the age between 20 and 40</p>
# 
# ## conclusion: Females have a higher rate of Survival (highest survival rate for those in 20-40 )
# 
# ## So we can say that Age, Sex are definitely related to Survival and we would have to account for this relationship when we try to account for the missing Age values later
# 
# <p>Enough about Age and Sex, let's also identify other possible relationships in the data df_train</p>

# In[16]:


df_train.info()


# In[17]:


df_train.head()


# # Port of Embarkation Vs Survival
# <p>Let's check out if port of embarkation (Embarked) is related to Survival rate</p>
# ##### Let's first view the distribution of Embarked in our data

# In[18]:


# C = Cherbourg, Q = Queenstown, S = Southampton
sns.factorplot(x='Embarked', data=df_train, kind='count')


# ##### Seems like most people embarked form Southampton. However, by the look of it, there are 2 missing values for Embarked, before imputing these values, lets view the exact rows with this missing value

# In[19]:


df_train[df_train['Embarked'].isnull()==True]


# looks like both passengers have the same ticket number and fare as well, they even share the same cabin!! Both even Survived!!
# 
# we can try to find passengers with similar profiles and see what their embarkation port is and then determine if we should just fill the embarked using the mode value because it seems very tempting to do so just because most passengers embarked from there

# In[20]:


sns.factorplot(x='Embarked', hue='Survived',data=df_train, kind='count')


# In[21]:


sns.factorplot(x='Embarked', hue='Sex',data=df_train, kind='count')


# from both plots above, we can see that the survival rate is higher for those who embarked from C (cherbourg) and Q (queenstown) compared to S(southampton) although most survivors still come from Southampton by sheer number of passengers who boarded there
# 
# We can also see that proportion of women from embarking from C and Q is much higher than S although we can't deny that most women still embarked from S 
# 
# ## Conclusion : survival rate is higher for those who embarked from C (cherbourg) and Q (queenstown) compared to S(southampton)
# 
# We can also try to see if fares have anything to do with port of embarkation 

# In[22]:


print('Min Fare: ',df_train.Fare.min())
print('Max Fare: ',df_train.Fare.max())
print('Mean Fare: ',df_train.Fare.mean())
sns.factorplot(x='Embarked', y='Fare', data=df_train, kind='point')


# In[23]:


print(df_train[df_train['Embarked'] == 'S'].Fare.mean())
print(df_train[df_train['Embarked'] == 'C'].Fare.mean())
print(df_train[df_train['Embarked'] == 'Q'].Fare.mean())


# In[24]:


sns.stripplot(data=df_train, x='Embarked', y='Fare', jitter=True, alpha=0.5)


# we can also combine a box plot and plot a swarmplot on top of it to view the distribution of fares for all 3 port of embarkation just by plotting both plots on the same grid (by putting them in the same cell) 

# In[25]:


sns.factorplot(x='Embarked', y='Fare', data=df_train, kind='box')
sns.swarmplot(data=df_train, x='Embarked', y='Fare', alpha=0.7, color='0.25')


# based on the plot above, we can conclude that on average, those who embarked from port C paid higher fares (surprisingly, those who embarked from this port also had a higher proportion of passengers who survived)
# 
# ### we can also impute the Embarkation port of the 2 passengers with C and justify that this is because both passengers have survived and they paid higher fares which is closer to the average fare of C
# 
# ## Conclusion: those who embarked from port C paid higher fares (surprisingly, those who embarked from this port also had a higher proportion of passengers who survived)
# 
# now, before we finally impute the missing values for Embarked for both passengers, let's try and also look at relationship between age and port of embarkation 

# In[26]:


sns.stripplot(data=df_train, x='Embarked', y='Age', jitter=True, alpha=0.5)


# not surprisingly, highest age group for ports S and C are within the 20-40 Age group (the same can actually be said for Q but the age distribution is a little sparse (spread out) for this port)
# 
# ## Conclusion: All Ports seems to have similar age distributions and as such, we may conclude that there may not be much of a correlation between Age and port of Embarkation 

# In[27]:


sns.stripplot(data=df_train, x='Embarked', y='Age', hue='Survived',jitter=True, alpha=0.5)


# In[28]:


sns.stripplot(data=df_train, x='Embarked', y='Age', hue='Sex',jitter=True, alpha=0.5)


# even if we included factors such as survival and sex, we can still see that there's not much of a difference in age and sex distribution for all ports and hence, we can safely say that there's not much of a correlation between age and sex with port of embarkation
# 
# 
# ## Conclusion: Not much of a correlation between age and sex with port of embarkation

# now let's see if there's a correlation between Age, Fare and Survival Rate !! 
# 
# Note that we now know that people who embarked from C paid higher fares on average but i we take a closer look at the stripplot above which shows fares paid versus por of embarkation, there's only 2 people in C which paid really really high fares. This could be an outlier as to why average fare paid in C is higher. Regardless, we do not want to overcomplicate the analysis for now and we will leave it at that. 
# 
# Now, we shall just focus on Fare and Survival Rate
# 
# # Fare vs Survival

# In[29]:


sns.jointplot(x='Age', y='Fare', data=df_train, alpha=0.5)


# it seems that from the plot above, most people paid less than $100 in fares. Let's confirm this with a distribution plot

# In[30]:


sns.distplot(a=df_train['Fare'], kde=False)


# so now we see that most people paid less than $100 in fares. How does this affect survival ? Is there a relationship?

# In[31]:


sns.stripplot(data=df_train,x='Survived', y='Fare', jitter=True, alpha=0.5)


# from the stripplot above, we can see that those who survived paid higher fares on average vs those who didn't
# 
# we can also view this from distribution plot by plotting distribution plot of those who survived above the distribution plot of the entire passenger population

# In[32]:


# this uses seaborn (plot the fare distribution for those who did not survive first)
sns.distplot(a=df_train[df_train['Survived'] == 0].Fare, kde=False)
# get the fares for only those who survived (plot them on top of those who did not survive)
sns.distplot(a=df_train[df_train['Survived'] == 1].Fare, kde=False)

# Higher green bars indicate that those who survived paid higher fares on average


# In[33]:


# this uses pandas' hist() method from Series object
df_train.groupby(df_train['Survived']).Fare.hist(alpha=0.5)


# we can also view the Fare statistics between those who survived and those who didn't

# In[34]:


df_train.groupby(df_train['Survived']).Fare.describe()


# we can see that those who survived has an average fare more than twice as much as those who didn't. Therefore, there's certainly a correlation between Fare and Survival rate
# 
# ## Conclusion: Those who paid higher fare have bigger chances of survival

# # Pclass vs Survival
# 
# let's see if there is any relationship between passenger class (Pclass) and Survival

# In[35]:


sns.factorplot(data=df_train ,x='Pclass', kind='count')
sns.factorplot(data=df_train, x='Survived', col='Pclass', kind='count')


# we see that most passengers are in pclass 3 but those in pclass 1 has higher survival rate and there are even more survivors that come from pclass 1 than 3. So there's certainly a correlation between pclass and survival rate
# 
# ## Conclusion: Passengers in pclass 1 are more likely to survive than those in other classes (about 50-50 for pclass 2)

# # Relationship between Fare and Pclass with Survival rate

# In[36]:


sns.swarmplot(data=df_train, x='Pclass', y='Fare', hue='Survived')


# In[37]:


sns.stripplot(data=df_train, x='Pclass', y='Fare', hue='Survived', jitter=True, alpha=0.5)


# In[38]:


df_train.groupby(['Pclass']).Fare.describe()


# ## Conclusion: Those in Pclass 1 paid higher fares on average and people who paid higher fares have higher survival rate

# # SibSp vs Survival
# 
# now, there's another column called sibsp and this is called sibling spouse and it represents the number of siblings and/or spouses that a passenger has on board with them
# 
# we would like to ascertain any relationship this has with survival rate of a passenger
# 
# let's first view the distribution

# In[39]:


df_train.head()


# In[40]:


df_train.SibSp.describe()


# In[41]:


sns.factorplot(data=df_train, x='SibSp', kind='count')


# seems like most people are alone. Let's view the distribution vs Survival
# 

# In[42]:


sns.factorplot(data=df_train, x='Survived', col='SibSp', kind='count')


# seems like most survivors have at most 1 SibSp (perhaps it helps that they did not need to worry or save other family members other than themselves) 
# 
# Surprisingly, those with SibSp 1 have a higher survival rate (50-50 for SibSp 2) and this dramatically reduces with increase in SibSp
# 
# SibSp definitely affect a passenger's survival rate 
# 
# ## Conclusion: Passengers with lower SibSp has higher survival rate

# # Parch vs Survival 

# let's do that same analysis for Parch since they essentially represent Parents and children on board with passengers (m=family members on board)

# In[43]:


print(df_train.info())
print('=================')
print(df_train.Parch.describe())
sns.factorplot(data=df_train, x='Parch', kind='count')


# In[44]:


sns.factorplot(data=df_train, x='Survived', col='Parch', kind='count')


# seems like most survivors have at most 1 Parch (perhaps it helps that they did not need to worry or save other family members other than themselves)
# 
# Surprisingly, those with Parch 1 have a higher survival rate (50-50 for Parch 2) and this dramatically reduces with increase in Parch
# 
# SibSp definitely affect a passenger's survival rate
# 
# ## Conclusion: Passengers with lower Parch has higher survival rate¶

# # Cabin vs Survival
# 
# Last but not least, we are going to analyse the relationship between cabin and survival rate 
# 
# now, we have left this till the end because there are just so many missing values for cabin (or maybe there's only 204 cabins on board and that the rest of the passengers do not have a cabin at all)

# In[45]:


print(df_train.Cabin.describe())


# In[46]:


# let's find out how many with cabin survived
df_train_w_cabin = df_train[df_train.Cabin.isnull() ==  False][['PassengerId', 'Survived', 'Cabin']]


# In[47]:


sns.factorplot(data=df_train_w_cabin, x='Survived', kind='count')


# from the plot above, we see that there's a higher proportion of those with cabins survived

# In[48]:


df_train_w_cabin.head()


# from the table above, we see that there are a few passengers with more than 1 cabin 

# In[49]:


df_train_w_cabin['NumOfCabin'] = df_train_w_cabin.Cabin.str.split(' ', n=0).apply(lambda x : len(x))


# now that we manage to split the cabins into arrays, we can get the number of cabins and add them to the table for us to see if there's any relationship between number of cabins and survival rate
# 
# after splitting the cabins into individual arrays, we can then count the length of each array to determine how many cabins does each passenger has

# In[50]:


df_train_w_cabin.head(n=10)


# now that we have added the NumOfCabins column to the dataframe, we can now analyse the relationship between this column and survival rate

# In[51]:


df_train_w_cabin.NumOfCabin.describe()


# In[52]:


sns.factorplot(data=df_train_w_cabin, x='NumOfCabin', kind='count')


# In[53]:


sns.factorplot(data=df_train_w_cabin, x='Survived', col='NumOfCabin', kind='count')


# wow, it looks like people with cabins actually had higher survival rates. As can be seen in every column, more people survived than perished
# 
# now, let us compare between those who have cabins and those who do not
# 
# ##### Note: For the purposes of comparison, we will assume that all passengers without cabins do not have a cabin. Therefore, they will have 0 as the number of cabins and the rest wilth cabins will be filled the their respective number of cabins

# In[54]:


# create new column NumOfCabins for df_train and first fill those with cabins first 
# and then impute null values with 0 later
df_train['NumOfCabin'] = df_train_w_cabin.NumOfCabin


# In[55]:


df_train.NumOfCabin.fillna(value=0, inplace=True)


# In[56]:


sns.factorplot(data=df_train, x='NumOfCabin', kind='count')


# In[57]:


sns.factorplot(data=df_train, col='NumOfCabin' ,x='Survived', kind='count')


# based on the plot above, it seems that those with cabins certainly has a higher rate of survival. We will certainly need to take this into account when we train our ML model.
# 
# ### Conclusion: Passengers with cabin(s) have a higher rate of survival

# # Part 2: Further Data analysis
# 
# After initially exploring the data, a few correlations were found and the data is then preprocessed accordingly, the results are in the range of 75-80% which is relatively low. So this part2 serves to further explore any relationships with survival rate which might have been unexplored earlier

# In[58]:


df_train.head()


# from the look of it, it seems that there are different types of tickets issued.
# 
# for the most part, these tickets are all in numbers but some contain letters as well. let's explore these

# In[59]:


df_train_copy = df_train.copy()


# In[60]:


def matchLetters(string):
    match = re.search('([a-zA-Z]+)', string)
    if(match):
        return 1
    else:
        return 0


# In[61]:


df_train_copy['Weird_Ticket'] = df_train_copy.Ticket.apply(lambda x : matchLetters(x)) 


# In[62]:


sns.factorplot(data=df_train_copy, col='Weird_Ticket', x='Survived', kind='count')


# now that we know which passengers have special ticket, we can see if there's a relationship between these special tickets and survival rate

# In[63]:


df_train_copy[df_train_copy.Ticket.duplicated()].head()

