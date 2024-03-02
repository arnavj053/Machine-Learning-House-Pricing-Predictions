#!/usr/bin/env python
# coding: utf-8

# In[45]:


# Initialize Otter
import otter
grader = otter.Notebook("projA1.ipynb")


# # Project A1: Exploring Cook County Housing
# 
# ## Due Date: Saturday, October 21st 11:59 PM PDT
# You must submit this assignment to Gradescope by the on-time deadline, Saturday, October at 11:59 PM. Please read the syllabus for the grace period policy. No late submissions beyond the grace period will be accepted. While course staff is happy to help you if you encounter difficulties with submission, we may not be able to respond to last-minute requests for assistance (TAs need to sleep, after all!). **We strongly encourage you to plan to submit your work to Gradescope several hours before the stated deadline.** This way, you will have ample time to reach out to staff for submission support. 
# 
# ### Collaboration Policy
# 
# Data science is a collaborative activity. While you may talk with others about the homework, we ask that you **write your solutions individually**. If you do discuss the assignments with others please **include their names** in the collaborators cell below.

# **Collaborators:** *list names here*

# ## Introduction
# 
# This project explores what can be learned from an extensive housing dataset that is embedded in a dense social context in Cook County, Illinois.
# 
# In project A1 (this assignment), we will guide you through some basic Exploratory Data Analysis (EDA) to understand the structure of the data. Next, you will be adding a few new features to the dataset, while cleaning the data as well in the process.
# 
# In project A2 (the following assignment), you will specify and fit a linear model for the purpose of prediction. Finally, we will analyze the error of the model and brainstorm ways to improve the model's performance.
# 
# 
# ## Grading
# Grading is broken down into autograded answers and free response. 
# 
# For autograded answers, the results of your code are compared to provided and/or hidden tests.
# 
# For free response, readers will evaluate how well you answered the question and/or fulfilled the requirements of the question.
# 
# Question | Manual | Points
# ----|----|----
# 1a | Yes | 1
# 1b | Yes | 1
# 1c | Yes | 1
# 1d | Yes | 1
# 2a | Yes | 1
# 2b | No | 1
# 3a | No | 1
# 3b | No | 1
# 3c | Yes | 1
# 4 | No | 2
# 5a | No | 1
# 5b | No | 2
# 5c | Yes | 2
# 6a | No | 1
# 6b | No | 2
# 6c | No | 2
# 6d | No | 1
# 7a | No | 1
# 7b | No | 2
# Total | 7 | 25

# In[46]:


import numpy as np

import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import zipfile
import os

# Plot settings
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['font.size'] = 12


# <br/><br/>
# <hr style="border: 5px solid #003262;" />
# <hr style="border: 1px solid #fdb515;" />
# 
# # The Data
# 
# The dataset consists of over 500,000 records from Cook County, Illinois, the county where Chicago is located. The dataset has 61 features in total; the 62nd is `Sale Price`, which you will predict with linear regression in the next part of this project. An explanation of each variable can be found in the included `codebook.txt` file (you can optionally open this by first clicking the `data` folder, then clicking `codebook.txt` file in the navigation pane). Some of the columns have been filtered out to ensure this assignment doesn't become overly long when dealing with data cleaning and formatting.
# 
# The data are split into training and test sets with 204,792 and 68,264 observations, respectively, but we will only be working on the training set for this part of the project.
# 
# Let's first extract the data from the `cook_county_data.zip`. Notice we didn't leave the `csv` files directly in the directory because they take up too much space without some prior compression. Just run the cells below: 

# In[47]:


with zipfile.ZipFile('data/cook_county_data.zip') as item:
    item.extractall()


# Let's load the training data.

# In[48]:


training_data = pd.read_csv("cook_county_train.csv", index_col='Unnamed: 0')


# As a good sanity check, we should at least verify that the data shape matches the description.

# In[49]:


# 204,792 observations and 62 features in training data
assert training_data.shape == (204792, 62)
# Sale Price is provided in the training data
assert 'Sale Price' in training_data.columns.values


# The next order of business is getting a feel for the variables in our data.  A more detailed description of each variable is included in `codebook.txt` (in the same directory as this notebook).  **You should take some time to familiarize yourself with the codebook before moving forward.**
# 
# Let's take a quick look at all the current columns in our training data.

# In[50]:


training_data.columns.values


# In[51]:


training_data['Description'][0]


# <br/><br/>
# <hr style="border: 1px solid #fdb515;" />
# 
# # Part 1: Contextualizing the Data
# 
# Let's try to understand the background of our dataset before diving into a full-scale analysis. 
# 
# **Note**: We will explore this dataset and its social context in greater detail in Lecture 15: Case Study (HCE). For now, we ask you to familiarize yourself with the overall structure of the dataset and the information it contains.

# <!-- BEGIN QUESTION -->
# 
# <br><br>
# 
# ---
# 
# ## Question 1a
# 
# Based on the columns in this dataset and the values that they take, what do you think each row represents? That is, what is the granularity of this dataset? 

# Based on the columns in this dataset and the values that they take, I think each row represents a property in Cook County. Each property has many attributes such as land square feet, town code, and more. 

# <!-- END QUESTION -->
# 
# <!-- BEGIN QUESTION -->
# 
# <br><br>
# 
# ---
# ## Question 1b
# Why do you think this data was collected? For what purposes? By whom?
# 
# This question calls for your speculation and is looking for thoughtfulness, not correctness. 

# I think this data was collected to make predictions about sale prices for properties in Cook County by analyzing various features such as the property class, neighborhood code, land square feet, wall material, and more. This would help develop a deeper understanding of the housing properties landscape in this county and see maybe what kinds of houses have the highest value, the density of number of households in particular neighborhoods, governmental policies like property taxes to be charged, etc. The data was most probably collected by a specific division of the county that is designated for these types of tasks. Furthermore, house realtors could have also been in charge of collecting this data. 

# <!-- END QUESTION -->
# 
# <!-- BEGIN QUESTION -->
# 
# <br><br>
# 
# ---
# ## Question 1c
# 
# Craft at least two questions about housing in Cook County that can be answered with this dataset and provide the type of analytical tool you would use to answer it (e.g. "I would create a ___ plot of ___ and ___" or "I would calculate the ___ [summary statistic] for ___ and ____"). Be sure to reference the columns that you would use and any additional datasets you would need to answer that question.

# I could create a linear regression model that analyzes the relationship between land square feet and sale price which can help in predicting house prices for how much land area they cover in Cook county (columns used would be 'Land Square Feet' and 'Sale Price'. Another question that I could answer using this dataset would be creating a histogram of the distribution of the ages of people that have houses in this county to see what the general demographic looks like ('Age' column used).

# ## <!-- END QUESTION -->
# 
# <!-- BEGIN QUESTION -->
# 
# <br><br>
# 
# ---
# ## Question 1d
# 
# Suppose now, in addition to the information already contained in the dataset, you also have access to several new columns containing demographic data about the owner, including race/ethnicity, gender, age, annual income, and occupation. Provide one new question about housing in Cook County that can be answered using at least one column of demographic data and at least one column of existing data and provide the type of analytical tool you would use to answer it.

# You could use the annual income column and neighborhood code column to analyze what the average annual income is in specific neighborhoods in the county and the analytical that could be used to model this relationship is a bar plot where you plot neighborhood column on the x-axis and their respective average annual incomes on the y-axis. 

# <!-- END QUESTION -->
# 
# <br/><br/>
# <hr style="border: 1px solid #fdb515;" />
# 
# # Part 2: Exploratory Data Analysis
# 
# This dataset was collected by the [Cook County Assessor's Office](https://datacatalog.cookcountyil.gov/Property-Taxation/Archive-Cook-County-Assessor-s-Residential-Sales-D/5pge-nu6u) in order to build a model to predict the monetary value of a home. You can read more about data collection in the CCAO’s [Residential Data Integrity Preliminary Report](https://gitlab.com/ccao-data-science---modeling/ccao_sf_cama_dev/-/blob/master/documentation/Preliminary%20Report%20on%20Data%20Integrity%20June%207,%202019.pdf). In Project A2, you will be building a linear regression model that predicts sales prices using training data, but it's important to first understand how the structure of the data informs such a model. In this section, we will make a series of exploratory visualizations and feature engineering in preparation for that prediction task.
# 
# Note that we will perform EDA on the **training data**.
# 
# ### Sale Price
# We begin by examining the distribution of our target variable `Sale Price`. We have provided the following helper method `plot_distribution` that you can use to visualize the distribution of the `Sale Price` using both the histogram and the box plot at the same time. Run the following 2 cells.

# In[52]:


def plot_distribution(data, label):
    fig, axs = plt.subplots(nrows=2)

    sns.distplot(
        data[label], 
        ax=axs[0]
    )
    sns.boxplot(
        x=data[label],
        width=0.3, 
        ax=axs[1],
        showfliers=False,
    )

    # Align axes
    spacer = np.max(data[label]) * 0.05
    xmin = np.min(data[label]) - spacer
    xmax = np.max(data[label]) + spacer
    axs[0].set_xlim((xmin, xmax))
    axs[1].set_xlim((xmin, xmax))

    # Remove some axis text
    axs[0].xaxis.set_visible(False)
    axs[0].yaxis.set_visible(False)
    axs[1].yaxis.set_visible(False)

    # Put the two plots together
    plt.subplots_adjust(hspace=0)
    fig.suptitle("Distribution of " + label)


# In[53]:


plot_distribution(training_data, label='Sale Price')


# At the same time, we also take a look at some descriptive statistics of this variable. Run the following cell.

# In[54]:


training_data['Sale Price'].describe()


# <!-- BEGIN QUESTION -->
# 
# <br><br>
# 
# ---
# ## Question 2a
# 
# Using the plots above and the descriptive statistics from `training_data['Sale Price'].describe()` in the cells above, identify one issue with the visualization above and briefly describe one way to overcome it. 

# One issue that I notice with the visualization above is the fact that the plot is not scaled properly and you can't read any of the important summary statistics created by the distplot or the boxplot functions. One way to overcome this is to use log function to spread out the data more properly so that it is comprehensible.

# <!-- END QUESTION -->
# 
# 

# In[55]:


# optional cell for scratch work


# <br><br>
# 
# ---
# ## Question 2b
# 
# To zoom in on the visualization of most households, we will focus only on a subset of `Sale Price` for this assignment. In addition, it may be a good idea to apply log transformation to `Sale Price`. In the cell below, reassign `training_data` to a new dataframe that is the same as the original one **except with the following changes**:
# 
# - `training_data` should contain only households whose price is at least $500.
# - `training_data` should contain a new `Log Sale Price` column that contains the log-transformed sale prices.
# 
# **You should NOT remove the original column `Sale Price` as it will be helpful for later questions.** If you accidentally remove it, just restart your kernel and run the cells again.
# 
# **Note**: This also implies from now on, our target variable in the model will be the log-transformed sale prices from the column `Log Sale Price`. 
# 
# *To ensure that any error from this part does not propagate to later questions, there will be no hidden tests for this question.*
# 
# 

# In[56]:


training_data = training_data[training_data['Sale Price'] >= 500]
training_data['Log Sale Price'] = np.log(training_data['Sale Price'])


# In[57]:


grader.check("q2b")


# Let's create a new distribution plot on the log-transformed sale price. As a sanity check, you should see that the distribution for the Log Scale Price is much more uniform.

# In[58]:


plot_distribution(training_data, label='Log Sale Price');


# <br><br>
# 
# ---
# ## Question 3a
# 
# 
# Is the following statement correct? Assign your answer to `q3statement`.
# 
#      "At least 25% of the properties in the training set sold for more than $200,000.00."
# 
# **Note:** The provided test for this question does not confirm that you have answered correctly; only that you have assigned each variable to `True` or `False`.
# 

# In[59]:


# This should be set to True or False
q3statement = True


# In[60]:


grader.check("q3a")


# <br><br>
# 
# ---
# ## Question 3b
# 
# Next, we want to explore if there is any correlation between `Log Sale Price` and the total area occupied by the property. The `codebook.txt` file tells us the column `Building Square Feet` should do the trick -- it measures "(from the exterior) the total area, in square feet, occupied by the building".
# 
# Let's also apply a log transformation to the `Building Square Feet` column.
# 
# In the following cell, create a new column `Log Building Square Feet` in our `training_data` that contains the log-transformed area occupied by each property. 
# 
# **You should NOT remove the original `Building Square Feet` column this time, as it will be used for later questions**. If you accidentally remove it, just restart your kernel and run the cells again.
# 
# *To ensure that any errors from this part do not propagate to later questions, there will be no hidden tests for this question.*
# 

# In[61]:


training_data['Log Building Square Feet'] = np.log(training_data['Building Square Feet'])


# In[62]:


grader.check("q3b")


# <!-- BEGIN QUESTION -->
# 
# <br><br>
# 
# ---
# ## Question 3c
# 
# In the visualization below, we created a `jointplot` with `Log Building Square Feet` on the x-axis, and `Log Sale Price` on the y-axis. In addition, we fit a simple linear regression line through the bivariate scatter plot in the middle.
# 
# Based on the following plot, would `Log Building Square Feet` make a good candidate as one of the features for our model? Why or why not?
# 
# **Hint:** To help answer this question, ask yourself: what kind of relationship does a “good” feature share with the target variable we aim to predict?
# 
# ![Joint Plot](images/q2p3_jointplot.png)
# 

# No, log building square feet doesn't seem to make a good candidate as one of the features for our model as there seem to be large number of points, specifically on the left bottom, that are quite far away from the regression line and there can definitely be a curve that fits this data better to minimize MSE.

# <!-- END QUESTION -->
# 
# <br><br>
# 
# ---
# ## Question 4
# 
# Continuing from the previous part, as you explore the dataset, you might still run into more outliers that prevent you from creating a clear visualization or capturing the trend of the majority of the houses. 
# 
# For this assignment, we will work to remove these outliers from the data as we run into them. Write a function `remove_outliers` that removes outliers from the dataset based off a threshold value of a variable. For example, `remove_outliers(training_data, 'Building Square Feet', lower=500, upper=8000)` should return a copy of `data` with only observations that satisfy `Building Square Feet` less than or equal to 8000 (inclusive) and `Building Square Feet` greater than 500 (exclusive).
# 
# **Note:** The provided tests simply check that the `remove_outliers` function you defined does not mutate the input data inplace. However, the provided tests do not check that you have implemented `remove_outliers` correctly so that it works with any data, variable, lower, and upper bound.

# In[63]:


def remove_outliers(data, variable, lower=-np.inf, upper=np.inf):
    """
    Input:
      data (DataFrame): the table to be filtered
      variable (string): the column with numerical outliers
      lower (numeric): observations with values lower than or equal to this will be removed
      upper (numeric): observations with values higher than this will be removed
    
    Output:
      a DataFrame with outliers removed
      
    Note: This function should not change mutate the contents of data.
    """  
    return data[(data[variable] > lower) & (data[variable] <= upper)]


# In[64]:


grader.check("q4")


# <br/><br/>
# <hr style="border: 1px solid #fdb515;" />
# 
# # Part 3: Feature Engineering
# 
# In this section, we will walk you through a few feature engineering techniques. 
# 
# ### Bedrooms
# 
# Let's start simple by extracting the total number of bedrooms as our first feature for the model. You may notice that the `Bedrooms` column doesn't actually exist in the original `DataFrame`! Instead, it is part of the `Description` column.
# 
# <br><br>
# 
# ---
# ## Question 5a
# 
# Let's take a closer look at the `Description` column first. Compare the description across a few rows together at the same time. For the following list of variables, how many of them can be extracted from the `Description` column? Assign your answer to a list of integers corresponding to the statements that you think are true (ie. `[1, 2, 3]`).
# 
# 1. The date the property was sold on.
# 2. The number of stories the property contains.
# 3. The previous owner of the property.
# 4. The address of the property.
# 5. The number of garages the property has.
# 6. The total number of rooms inside the property.
# 7. The total number of bedrooms inside the property.
# 8. The total number of bathrooms inside the property.

# In[65]:


# optional cell for scratch work 


# In[66]:


q5a = [1, 2, 4, 6, 7, 8]


# In[67]:


grader.check("q5a")


# In[68]:


# optional cell for scratch work


# <br><br>
# 
# ---
# ## Question 5b
# 
# Write a function `add_total_bedrooms(data)` that returns a copy of `data` with an additional column called `Bedrooms` that contains the total number of bedrooms (**as integers**) for each house. Treat missing values as zeros, if necessary. Remember that you can make use of vectorized code here; you shouldn't need any `for` statements. 
# 
# **Hint**: You should consider inspecting the `Description` column to figure out if there is any general structure within the text. Once you have noticed a certain pattern, you are set with the power of Regex!
# 

# In[69]:


import re
def add_total_bedrooms(data):
    """
    Input:
      data (DataFrame): a DataFrame containing at least the Description column.

    Output:
      a Dataframe with a new column "Bedrooms" containing ints.

    """
    with_rooms = data.copy()
    bedroom_counts = data['Description'].str.extract(r'(\d)+ of which are bedrooms')[0].fillna(0).astype(int)
    with_rooms['Bedrooms'] = bedroom_counts
    return with_rooms

training_data = add_total_bedrooms(training_data)


# In[70]:


grader.check("q5b")


# <!-- BEGIN QUESTION -->
# 
# <br><br>
# 
# ---
# ## Question 5c
# 
# Create a visualization that clearly and succinctly shows if there exists an association between  `Bedrooms` and `Log Sale Price`. A good visualization should satisfy the following requirements:
# - It should avoid overplotting.
# - It should have clearly labeled axes and a succinct title.
# - It should convey the strength of the correlation between `Sale Price` and the number of rooms: in other words, you should be able to look at the plot and describe the general relationship between `Log Sale Price` and `Bedrooms`
# 
# **Hint**: A direct scatter plot of the `Sale Price` against the number of rooms for all of the households in our training data might risk overplotting.
# 

# In[71]:


sns.boxplot(x ='Bedrooms', y = 'Log Sale Price', data = training_data)
plt.xlabel('Bedrooms')
plt.ylabel('Log Sale Price')
plt.title('Association between # of Bedrooms and Sale Prices')
plt.show()


# <!-- END QUESTION -->
# 
# Now, let's take a look at the relationship between neighborhood and sale prices of the houses in our dataset.
# Notice that currently we don't have the actual names for the neighborhoods. Instead we will use a similar column `Neighborhood Code` (which is a numerical encoding of the actual neighborhoods by the Assessment office).

# <br><br>
# 
# ---
# ## Question 6a
# 
# Before creating any visualization, let's quickly inspect how many different neighborhoods we are dealing with.
# 
# Assign the variable `num_neighborhoods` with the total number of unique neighborhoods in `training_data`. 
# 

# In[72]:


num_neighborhoods = len(training_data['Neighborhood Code'].unique())
num_neighborhoods


# In[73]:


grader.check("q6a")


# <br><br>
# 
# ---
# ## Question 6b
# 
# If we try directly plotting the distribution of `Log Sale Price` for all of the households in each neighborhood using the `plot_categorical` function from the next cell, we would get the following visualization.
# 
# 
# ![overplot](images/q5p2_catplot.png)
# 

# In[74]:


def plot_categorical(neighborhoods):
    fig, axs = plt.subplots(nrows=2)

    sns.boxplot(
        x='Neighborhood Code',
        y='Log Sale Price',
        data=neighborhoods,
        ax=axs[0],
    )

    sns.countplot(
        x='Neighborhood Code',
        data=neighborhoods,
        ax=axs[1],
    )

    # Draw median price
    axs[0].axhline(
        y=training_data['Log Sale Price'].median(), 
        color='red',
        linestyle='dotted'
    )

    # Label the bars with counts
    for patch in axs[1].patches:
        x = patch.get_bbox().get_points()[:, 0]
        y = patch.get_bbox().get_points()[1, 1]
        axs[1].annotate(f'{int(y)}', (x.mean(), y), ha='center', va='bottom')

    # Format x-axes
    axs[1].set_xticklabels(axs[1].xaxis.get_majorticklabels(), rotation=90)
    axs[0].xaxis.set_visible(False)

    # Narrow the gap between the plots
    plt.subplots_adjust(hspace=0.01)


# Oh no, looks like we have run into the problem of overplotting again! 
# 
# You might have noticed that the graph is overplotted because **there are actually quite a few neighborhoods in our dataset**! For the clarity of our visualization, we will have to zoom in again on a few of them. The reason for this is our visualization will become quite cluttered with a super dense x-axis.
# 
# Assign the variable `in_top_20_neighborhoods` to a copy of `training_data` that has been filtered to only contain rows corresponding to properties that are in one of the top 20 most populous neighborhoods. We define the “top 20 neighborhoods” as being the 20 neighborhood codes that have the greatest number of properties within them.
# 

# In[75]:


top_20_neighborhood_codes = training_data['Neighborhood Code'].value_counts().sort_values(ascending = False).head(20).reset_index()['Neighborhood Code']
in_top_20_neighborhoods = training_data[training_data['Neighborhood Code'].isin(top_20_neighborhood_codes)]
in_top_20_neighborhoods


# In[76]:


grader.check("q6b")


# Let's create another of the distribution of sale price within in each neighborhood again, but this time with a narrower focus!

# In[77]:


plot_categorical(neighborhoods=in_top_20_neighborhoods)


# <br><br>
# 
# ---
# ## Question 6c
# 
# From the plot above, we can see that there is much less data available for some neighborhoods. For example, Neighborhood 71 has only around 27% of the number of datapoints as Neighborhood 30.
# 
# One way we can deal with the lack of data from some neighborhoods is to create a new feature that bins neighborhoods together. We’ll categorize our neighborhoods in a crude way. In Question 6c, we’ll compute how “expensive” each neighborhood is by aggregating the `Log Sale Price`s for all properties in a particular neighborhood using a `metric`, such as the median. We’ll use this `metric` to find the top `n` most expensive neighborhoods. Then, in Question 6d, we’ll label these “expensive neighborhoods” and leave all other neighborhoods unmarked.
# 
# Fill in `find_expensive_neighborhoods` to return a **list** of the neighborhood codes of the **top `n`** most expensive neighborhoods as measured by our choice of aggregating function, `metric`.
# 
# For example, calling `find_expensive_neighborhoods(training_data, n=3, metric=np.median)` should return the 3 neighborhood codes with the highest median `Log Sale Price` computed across all properties in those neighborhood codes. 
# 
# 

# In[78]:


def find_expensive_neighborhoods(data, n=3, metric=np.median):
    """
    Input:
      data (DataFrame): should contain at least a int-valued 'Neighborhood Code'
        and a numeric 'Log Sale Price' column
      n (int): the number of top values desired
      metric (function): function used for aggregating the data in each neighborhood.
        for example, np.median for median prices
    
    Output:
      a list of the the neighborhood codes of the top n highest-priced neighborhoods 
      as measured by the metric function
    """
    neighborhoods = data.groupby('Neighborhood Code')['Log Sale Price'].agg(metric).sort_values(ascending = False).head(n).index
    
    # This makes sure the final list contains the generic int type used in Python3, not specific ones used in numpy.
    return [int(code) for code in neighborhoods]

expensive_neighborhoods = find_expensive_neighborhoods(training_data, 3, np.median)
expensive_neighborhoods


# In[79]:


grader.check("q6c")


# <br><br>
# 
# ---
# ## Question 6d
# 
# We now have a list of neighborhoods we've deemed as higher-priced than others.  Let's use that information to write an additional function `add_expensive_neighborhood` that takes in a `DataFrame` of housing data (`data`) and a list of neighborhood codes considered to be expensive (`expensive_neighborhoods`). You can think of `expensive_neighborhoods` as being the output of the function `find_expensive_neighborhoods` from Question 6c. 
# 
# Using these inputs, `add_expensive_neighborhood` should add a column to `data` named `in_expensive_neighborhood` that takes on the **integer** value of 1 if a property is part of a neighborhood in `expensive_neighborhoods` and the integer value of 0 if it is not. This type of variable is known as an **indicator variable**.
# 
# **Hint:** [`pd.Series.astype`](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.Series.astype.html) may be useful for converting `True`/`False` values to integers.
# 

# In[80]:


def add_in_expensive_neighborhood(data, expensive_neighborhoods):
    """
    Input:
      data (DataFrame): a DataFrame containing a 'Neighborhood Code' column with values
        found in the codebook
      neighborhoods (list of strings): strings should be the names of neighborhoods
        pre-identified as expensive
    Output:
      DataFrame identical to the input with the addition of a binary
      in_expensive_neighborhood column
    """
    data['in_expensive_neighborhood'] = data['Neighborhood Code'].isin(expensive_neighborhoods).astype(int)
    return data

expensive_neighborhoods = find_expensive_neighborhoods(training_data, 3, np.median)
training_data = add_in_expensive_neighborhood(training_data, expensive_neighborhoods)


# In[81]:


grader.check("q6d")


# In the following question, we will take a closer look at the `Roof Material` feature of the dataset and examine how we can incorporate categorical features into our linear model.

# <br><br>
# 
# ---
# ## Question 7a
# 
# If we look at `codebook.txt` carefully, we can see that the Assessor's Office uses the following mapping for the numerical values in the `Roof Material` column.
# ```
# Roof Material (Nominal): 
# 
#        1    Shingle/Asphalt
#        2    Tar & Gravel
#        3    Slate
#        4    Shake
#        5    Tile
#        6    Other
# ```
# 
# Write a function `substitute_roof_material` that replaces each numerical value in `Roof Material` with their corresponding roof material. Your function should return a new `DataFrame`, not modify the existing `DataFrame`.
# 
# **Hint**: the `DataFrame.replace` ([documentation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.replace.html)) method may be useful here.
# 
# 

# In[82]:


def substitute_roof_material(data):
    """
    Input:
      data (DataFrame): a DataFrame containing a 'Roof Material' column.  Its values
                         should be limited to those found in the codebook
    Output:
      DataFrame identical to the input except with a refactored 'Roof Material' column
    """
    data = data.copy()
    data['Roof Material'] = data['Roof Material'].replace({1: 'Shingle/Asphalt', 2: 'Tar & Gravel', 3: 'Slate', 4: 'Shake', 5: 'Tile', 6: 'Other'})
    return data
    
training_data_mapped = substitute_roof_material(training_data)
training_data_mapped.head()


# In[83]:


grader.check("q7a")


# <br><br>
# 
# ---
# ## Question 7b
# 
# #### An Important Note on One Hot Encoding 
# 
# Unfortunately, simply replacing the integers with the appropriate strings isn’t sufficient for using `Roof Material` in our model.  Since `Roof Material` is a categorical variable, we will have to one-hot-encode the data. For more information on why we want to use one-hot-encoding, refer to this [link](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/).
# 
# Complete the following function `ohe_roof_material` that returns a `DataFrame` with the new column one-hot-encoded on the roof material of the household. These new columns should have the form `Roof Material_MATERIAL`. Your function should return a new `DataFrame` and **should not modify the existing `DataFrame`**.
# 
# You should use Scikit-learn’s `OneHotEncoder` to perform the one-hot-encoding. `OneHotEncoder` will automatically generate column names of the form `Roof Material_MATERIAL`. Refer back to the video walkthrough for Question 1 of Lab 7 for an example of its use. Note that unlike in the lab example, in this problem we only wish to construct the one-hot-encoding columns **without removing any columns**.

# In[88]:


from sklearn.preprocessing import OneHotEncoder

def ohe_roof_material(data):
    """
    One-hot-encodes roof material. New columns are of the form "Roof Material_MATERIAL".
    """
    oh_enc = OneHotEncoder()
    roof_data = oh_enc.fit_transform(data[['Roof Material']]).toarray()
    roof_df = pd.DataFrame(data = roof_data, columns = oh_enc.get_feature_names_out(), index = data.index)
    return data.join(roof_df)
training_data_ohe = ohe_roof_material(training_data_mapped)
# This line of code will display only the one-hot-encoded columns in training_data_ohe that 
# have names that begin with “Roof Material_" 
training_data_ohe.filter(regex='^Roof Material_').head(10)


# In[166]:


grader.check("q7b")


# ## Submission
# 
# Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output. The cell below will generate a zip file for you to submit. **Please save before exporting!**
# 
# After you have run the cell below and generated the zip file, you can open the PDF <a href='projA1.pdf' download>here</a>.

# In[43]:


# Save your notebook first, then run this cell to export your submission.
grader.export(run_tests=True)


#  
