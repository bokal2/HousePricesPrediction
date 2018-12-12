
# coding: utf-8

# ## Imports
# ** Import pandas, numpy, matplotlib,and seaborn. Then set %matplotlib inline 
# (we'll import sklearn as we need it.)**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data
# 
# We'll work with the Ecommerce Customers csv file from the company. It has Customer info, suchas Email, Address, and their color Avatar. Then it also has numerical value columns:
# 
# * Avg. Session Length: Average session of in-store style advice sessions.
# * Time on App: Average time spent on App in minutes
# * Time on Website: Average time spent on Website in minutes
# * Length of Membership: How many years the customer has been a member. 

# In[276]:


customers = pd.read_csv("Ecommerce Customers")


# **Let's check the head of customers, and check out its info() and describe() methods.**

# In[277]:


customers.head()


# In[278]:


customers.describe()


# In[279]:


customers.info()


# ## Exploratory Data Analysis
# 
# **Let's explore the data!**
# 
# We'll only be using the numerical data of the csv file.
# ___
# **Will use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?**

# In[280]:


sns.set_palette("GnBu_d")
sns.set_style('whitegrid')


# In[281]:


# More time on site, more money spent.
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)


# ** We will do the same but with the Time on App column instead. **

# In[282]:


sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)


# ** Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.**

# In[283]:


sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)


# **Let's explore these types of relationships across the entire data set. Use [pairplot](https://stanford.edu/~mwaskom/software/seaborn/tutorial/axis_grids.html#plotting-pairwise-relationships-with-pairgrid-and-pairplot) to recreate the plot below.

# In[284]:


sns.pairplot(customers)


# **Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?**

# In[285]:


# Length of Membership 


# **We'll create a linear model plot (using seaborn's lmplot) of  Yearly Amount Spent vs. Length of Membership. **

# In[286]:


sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)


# ## Training and Testing Data
# 
# Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.
# ** We set variable X equal to the numerical features of the customers and variable y equal to the "Yearly Amount Spent" column. **

# In[287]:


y = customers['Yearly Amount Spent']


# In[288]:


X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]


# ** We use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101**

# In[289]:


from sklearn.model_selection import train_test_split


# In[290]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ## Training the Model
# 
# Now its time to train our model on our training data!
# 
# ** Importing LinearRegression from sklearn.linear_model **

# In[291]:


from sklearn.linear_model import LinearRegression


# **Creating an instance of a LinearRegression() model named lm.**

# In[292]:


lm = LinearRegression()


# ** Train/fit lm on the training data.**

# In[293]:


lm.fit(X_train,y_train)


# **Printing out the coefficients of the model**

# In[294]:


# The coefficients
print('Coefficients: \n', lm.coef_)


# ## Predicting Test Data
# Now that we have fit our model, let's evaluate its performance by predicting off the test values!
# 
# ** We use lm.predict() to predict off the X_test set of the data.**

# In[295]:


predictions = lm.predict( X_test)


# ** Creating a scatterplot of the real test values versus the predicted values. **

# In[296]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# ## Evaluating the Model
# 
# Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).
# 
# ** Calculating the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.

# In[303]:


# calculate these metrics by hand!
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# ## Residuals
# 
# Now that we have gotten a very good model with a good fit. Let's quickly explore the residuals to make sure everything was okay with our data. 
# 
# ** Lets's plot a histogram of the residuals and make sure it looks normally distributed. We will use either seaborn distplot, or just plt.hist().**

# In[317]:


sns.distplot((y_test-predictions),bins=50);


# ## Conclusion
# We still want to figure out the answer to the original question, do we focus our efforst on mobile app or website development? Or maybe that doesn't even really matter, and Membership Time is what is really important.  Let's see if we can interpret the coefficients at all to get an idea.
# 
# ** We will recreate the dataframe below. **

# In[298]:


coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# ** How can we interpret these coefficients? **

# Interpreting the coefficients:
# 
# - Holding all other features fixed, a 1 unit increase in **Avg. Session Length** is associated with an **increase of 25.98 total dollars spent**.
# - Holding all other features fixed, a 1 unit increase in **Time on App** is associated with an **increase of 38.59 total dollars spent**.
# - Holding all other features fixed, a 1 unit increase in **Time on Website** is associated with an **increase of 0.19 total dollars spent**.
# - Holding all other features fixed, a 1 unit increase in **Length of Membership** is associated with an **increase of 61.27 total dollars spent**.

# **Do you think the company should focus more on their mobile app or on their website?**

# 
# This is tricky, there are two ways to think about this: Develop the Website to catch up to the performance of the mobile app, or develop the app more since that is what is working better. This sort of answer really depends on the other factors going on at the company, we would probably want to explore the relationship between Length of Membership and the App or the Website before making our conclusion!
# 
