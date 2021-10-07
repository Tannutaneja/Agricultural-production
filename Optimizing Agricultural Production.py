#!/usr/bin/env python
# coding: utf-8

# In[3]:


# For manipulations
import numpy as np
import pandas as pd

#For data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

#For interactivity
from ipywidgets import interact


# In[4]:


data = pd.read_csv('Agriculture_Data.csv')


# In[5]:


data.describe( include = 'all')


# In[6]:


print("Shape of the dataset is:", data.shape)


# In[7]:


data.head()


# In[8]:


data.isnull().sum()


# In[9]:


data['label'].value_counts()

# print(a.dtype)
# data = list(a.items())
# a = np.array(data)
# print(a.dtype,"\n",a[:,1])


# In[10]:


avg_N = format(data['N'].mean())
avg_P = format(data['P'].mean())
avg_K = format(data['K'].mean())
avg_temp = format(data['temperature'].mean())
avg_humidity = format(data['humidity'].mean())
avg_rainfall = format(data['rainfall'].mean())
avg_ph = format(data['ph'].mean())


# In[11]:


print('average Nitrogen content is:', avg_N)
print('average Phosphorous content is:', avg_P)
print('average Potassium content is:', avg_K)
print('average Temperature in celcius is:', avg_temp)
print('average relative Humidity in mm is:', avg_humidity)
print('average rainfall in mm is:', avg_rainfall)
print('average ph of the soil is:', avg_ph)


# In[12]:


data[data['N'] == 90]


# In[13]:


#Let's check the summary Statistics for each of the crops

@interact

def summary(crops = list(data['label'].value_counts().index)):
    x = data[data['label'] == crops]
    #print(x)
    print("--------------------")
    print("Statistics for Nitrogen")
    print("Minimum Nitrogen required:", x['N'].min())
    print("Average Nitrogen required:", x['N'].mean())
    print("Maximum Nitrogen required:", x['N'].max())

    print("--------------------")
    print("Statistics for Phosphorous")
    print("Minimum Phosphorous required:", x['P'].min())
    print("Average Phosphorous required:", x['P'].mean())
    print("Maximum Phosphorous required:", x['P'].max())

    
    print("--------------------")
    print("Statistics for Potassium")
    print("Minimum Potassium required:", x['K'].min())
    print("Average Potassium required:", x['K'].mean())
    print("Maximum Potassium required:", x['K'].max())


    print("--------------------")
    print("Statistics for Temperature")
    print("Minimum Temperature required:", x['temperature'].min())
    print("Average Temperature required:", x['temperature'].mean())
    print("Maximum Temperature required:", x['temperature'].max())
    
    
    print("--------------------")
    print("Statistics for Humidity")
    print("Minimum Humidity required:", x['humidity'].min())
    print("Average Humidity required:", x['humidity'].mean())
    print("Maximum Humidity required:", x['humidity'].max())
    
    
    print("--------------------")
    print("Statistics for ph")
    print("Minimum ph required:", x['ph'].min())
    print("Average ph required:", x['ph'].mean())
    print("Maximum ph required:", x['ph'].max())
    
    
    print("--------------------")
    print("Statistics for rainfall")
    print("Minimum rainfall required:", x['rainfall'].min())
    print("Average rainfall required:", x['rainfall'].mean())
    print("Maximum rainfall required:", x['rainfall'].max())


# In[14]:


data['N']


# In[15]:


@interact

def compare(conditions = {'N','P','K','temperature','ph','humidity','rainfall'}):
    print("Average value for", conditions,"is {0:.2f}".format(data[conditions].mean()))
    print("---------------------------------------------------------------------------")
    print("Rice : {0:.2f}".format(data[data['label'] == 'rice'][conditions].mean()))
    print("Black Grams : {0:.2f}".format(data[data['label'] == 'blackgram'][conditions].mean()))
    print("Banana : {0:.2f}".format(data[data['label'] == 'banana'][conditions].mean()))
    print("Jute : {0:.2f}".format(data[data['label'] == 'jute'][conditions].mean()))
    print("Coconut : {0:.2f}".format(data[data['label'] == 'coconut'][conditions].mean()))
    print("Apple : {0:.2f}".format(data[data['label'] == 'apple'][conditions].mean()))
    print("Papaya : {0:.2f}".format(data[data['label'] == 'papaya'][conditions].mean()))
    print("Muskmelon : {0:.2f}".format(data[data['label'] == 'muskmelon'][conditions].mean()))
    print("Grapes : {0:.2f}".format(data[data['label'] == 'grapes'][conditions].mean()))
    print("Watermelon : {0:.2f}".format(data[data['label'] == 'watermelon'][conditions].mean()))
    print("Kidney Beans : {0:.2f}".format(data[data['label'] == 'kidneybeans'][conditions].mean()))
    print("Mung Beans : {0:.2f}".format(data[data['label'] == 'mungbean'][conditions].mean()))
    print("Oranges : {0:.2f}".format(data[data['label'] == 'orange'][conditions].mean()))
    print("Chick Peas : {0:.2f}".format(data[data['label'] == 'chickpea'][conditions].mean()))
    print("Lentils : {0:.2f}".format(data[data['label'] == 'lentil'][conditions].mean()))
    print("Cotton : {0:.2f}".format(data[data['label'] == 'cotton'][conditions].mean()))
    print("Maize : {0:.2f}".format(data[data['label'] == 'maize'][conditions].mean()))
    print("Moth Beans : {0:.2f}".format(data[data['label'] == 'mothbeans'][conditions].mean()))
    print("Pigeon Peas : {0:.2f}".format(data[data['label'] == 'pigeonpeas'][conditions].mean()))
    print("Mango : {0:.2f}".format(data[data['label'] == 'mango'][conditions].mean()))
    print("Pomegranate : {0:.2f}".format(data[data['label'] == 'pomegranate'][conditions].mean()))
    print("Coffee : {0:.2f}".format(data[data['label'] == 'coffee'][conditions].mean()))


# In[16]:


@interact
def compare(conditions = ['N','P','K','temperature','ph','humidity','rainfall']):
    print("Crops which require greater than average", conditions, '\n')
    print(data[data[conditions] > data[conditions].mean()]['label'].unique())
    print("-------------------------------------------------------------")
    print("Crops which require less than averge",conditions, '\n')
    print(data[data[conditions] <= data[conditions].mean()]['label'].unique())


# In[17]:


plt.subplot(2,4,1)
sns.distplot(data['N'],color = 'darkblue')
plt.xlabel("Ratio of Nitrogen",fontsize = 12)
plt.grid()

plt.subplot(2,4,2)
sns.distplot(data['P'],color = 'darkblue')
plt.xlabel("Ratio of Phosphorus",fontsize = 12)
plt.grid()

plt.subplot(2,4,3)
sns.distplot(data['K'],color = 'darkblue')
plt.xlabel("Ratio of Potassium",fontsize = 12)
plt.grid()

plt.subplot(2,4,4)
sns.distplot(data['temperature'],color = 'darkblue')
plt.xlabel("Ratio of temperature",fontsize = 12)
plt.grid()

plt.subplot(2,4,5)
sns.distplot(data['rainfall'],color = 'darkblue')
plt.xlabel("Ratio of Rainfall",fontsize = 12)
plt.grid()

plt.subplot(2,4,6)
sns.distplot(data['humidity'],color = 'darkblue')
plt.xlabel("Ratio of Humidity",fontsize = 12)
plt.grid()

plt.subplot(2,4,7)
sns.distplot(data['ph'],color = 'darkblue')
plt.xlabel("Ratio of PH",fontsize = 12)
plt.grid()



plt.suptitle('Distribution for Agricultural Conditions', fontsize = 20)
plt.show()


# In[18]:


print("Some interesting patterns")
print("-----------------------------")
print("Crops which requires very high ratio of Nitrogen content in soil:", data[data['N'] > 120]['label'].unique())
print("Crops which requires very high ratio of Phosphorous content in soil:", data[data['P'] > 100]['label'].unique())
print("Crops which requires very high ratio of Potassium content in soil:", data[data['K'] > 200]['label'].unique())
print("Crops which requires very high Rainfall",data[data['rainfall']>200]['label'].unique())
print("Crops which requires very Low Temperature",data[data['temperature']<10]['label'].unique())
print("Crops which requires very high Temperature",data[data['temperature']>40]['label'].unique())
print("Crops which requires very low Humidity",data[data['humidity']<20]['label'].unique())
print("Crops which requires very low PH",data[data['ph']<4]['label'].unique())
print("Crops which requires very high PH",data[data['ph']>9]['label'].unique())


# In[19]:


### lets understand which crop will be grown in summer season, winter season, rainy season


print("Summer crops")
print(data[(data['temperature'] > 30) & (data['humidity'] > 50)]['label'].unique())
print('-------------------------------------')

print("Winter crops")
print(data[(data['temperature'] < 20) & (data['humidity'] > 30)]['label'].unique())
print('-------------------------------------')

print("Rainy Crops")
print(data[(data['rainfall'] > 200) & (data['humidity'] > 30)]['label'].unique())


# In[20]:


from sklearn.cluster import KMeans

#removing the labels column
x = data.drop(['label'], axis=1)
print(x)
#selecting all the values of the data
x=x.values
print(x.shape,"\n",x)


# In[21]:


# let determine the optimum number of clusters within the dataset 

plt.rcParams['figure.figsize']=(10,4)

wcss=[]
for i in range(1,11):
    km = KMeans(n_clusters=i, init='k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km.fit(x)
    wcss.append(km.inertia_)


# In[22]:


#lets plot the results
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No.of Clusters')
plt.ylabel('wcss')
plt.show()


# In[23]:


#lets implement the K Means algorthim to perform clustering analysis
km = KMeans(n_clusters = 4, init='k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

a = data['label']
y_means = pd.DataFrame(y_means)
z = pd.concat([y_means, a], axis=1)
z = z.rename(columns = {0: 'cluster'})

#lets check the clusters of each crops
print('Lets check the results after applying the k means clustering analysis \n')
print('Crops in first cluster:',z[z['cluster'] == 0]['label'].unique())
print('---------------------------------------------------------------')

print('Crops in second cluster:',z[z['cluster'] == 1]['label'].unique())
print('---------------------------------------------------------------')

print('Crops in third cluster:',z[z['cluster'] == 2]['label'].unique())
print('---------------------------------------------------------------')

print('Crops in forth cluster:',z[z['cluster'] == 3]['label'].unique())


# In[24]:


print(z)


# In[25]:


# Lets split the Dataset for predictive modeling

y = data['label']
x = data.drop(['label'], axis = 1)

print('Shape of x:',x.shape)
print('Shape of y:',y.shape)


# In[26]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

print("The shape of x_train:",x_train.shape)
print("The shape of x_test:",x_test.shape)
print("The shape of y_train:",y_train.shape)
print("The shape of y_test:",y_test.shape)


# In[27]:


# Lets create predictive model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

model = lda()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


# In[28]:


cd = model.score(x_test,y_test)
print(cd)


# In[31]:


# Lets evaluate the Model performance

from sklearn.metrics import confusion_matrix

# Lets print the confusion matrix first
plt.rcParams['figure.figsize'] = (5,5)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot = True, cmap = 'Wistia')

plt.show()


# In[33]:


# Lets print the classification report also
from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)
print(cr)


# In[34]:


# Lets check the head of the dataset
data.head()


# In[35]:


prediction = model.predict((np.array([[90,40,40,20,80,7,200]])))
print("The suggested crop for the given climatic conditon is: ",prediction)


# In[ ]:





# In[ ]:




