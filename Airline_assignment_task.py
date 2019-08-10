#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Assignment solution_


# In[ ]:


#Airline_case_study


# In[ ]:


Ans 1.1 
In machine learning we need trainig dataset like we extract the data from the dataset.So 
X =  [['priority boarding','extra leg room','food and beverages']]
y =  ['exact seating'] # here the data will be in categorial data like if he want the exact sitting  
                          #  1  means  yes
                           # 0  means No  
Here,in Regression Problem We can use Linear Regression, so that we can predict the outcome very easily and accurate.
Here,In this we need to find out the evalauation Metrix like we can go for further approach like

For classification problems, we have only used classification accuracy for evaluation 

Mean Absolute Error (MAE) is the mean of the absolute value of the errors:

1n∑i=1n|yi−y^i|
 
Mean Squared Error (MSE) is the mean of the squared errors:

1n∑i=1n(yi−y^i)2
 
Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors:

1n∑i=1n(yi−y^i)2−−− √
    
Ans 1.2  The purchase rate for priority boarding is 20 % while for food is 2 % and here in this case we need to handle 
         some other ways like if we purchase rate of boarding is higher than the food so, we need to decrese the food 
         prices and after that purchaging rate slighly increase with the decrease in food prices. 


Ans 1.3 Yes,the concern is valid as a data scientist,because if we retrain our model again and again then some sort
        of error will occor and at the end we did not  get a desired output but if we tuned our model like apply 
        machine learning algorithm SVM then we get a chance to improve our training data and accuracy of the model
        will also increase.


# In[ ]:


Ans 2.1 Yes in machine learning if we got a  99.99% accuracy then model is highly trained and we get a good
        prediction result and overfitting of the data is not there and we get a good prediction value.
     
         confusion matrix:-           Actual 
                                  BAD       Not BAd
        predicted         BAd     10         99,990      
                        Not BAD   99,990     10        
    
         precision -    10+10/10+99,990+10+999,990
                          
                          20/200000
                             0.001 
                 

Ans 2.2   In,logistic regression is a predictive analysis,and  it is used to describe data 
          and to explain the relationship between one dependent binary variable and independent variable.
          Here, In logistic Regression the binary outcome value is(0,1), so in logistic Sigmoid graph(curve) is there in 
          this graph goes from 0 and 1 and in between we set a threshold value. 
                    
        2.2.1  The amount that the weights are updated during training is referred to as the step size or the “learning rate.” 
               Specifically, the learning rate is a configurable hyperparameter used in the training of neural networks that
               has a small positive value, often in the range between 0.0 and 1.0.
                
        2.2.2  The model which uses L2 is called Ridge Regression. The key difference between these L1 and L2  is the penalty term. 
                Ridge regression adds “squared magnitude” of coefficient as penalty term to the loss function.
                L2 regularization adds a penalty equal to the sum of the squared value of the coefficients. 
                The L2 regularization will force the parameters to be relatively small, the bigger the penalization.
                
       2.2.3.   The batch size is a number of samples processed before the model is updated. The number of epochs is 
                 the number of complete passes through the training dataset. 
        
       2.2.4.   Here in this case the predicted probabilty is 1  and the probability is not greater than 1 so,
                we the output we get in betwenn 0 and 1 only. 


# In[ ]:


Ans 3.1 Here in this question to calculate probability we use Naive Bayes machine algorithm because Naive Bayes is "probabilistic classifiers" 
        so we can get a better result using Naive Bayes.


# In[ ]:


Ans 4.1
# import the libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# load the dataset from the csv file using pandas
dataset = pd.read_csv("/home/negi/negi/Machine_Learn/ML/New folder/data/diabetes.csv", header =0)
print("dataset \n",dataset)


# In[3]:


#only 5 will show in the dataset
dataset.head()


# In[4]:


#explore the dataset
dataset.describe()


# In[5]:


# explore the dataset
dataset.info()


# In[6]:


# taking columns from the dataset
feature_colu = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']


# In[7]:


feature_colu


# In[ ]:


#visualization


# In[9]:


sns.jointplot(x='Glucose',y='BloodPressure',data=dataset,kind='scatter')


# In[10]:


sns.distplot(dataset['Insulin'])


# In[11]:


sns.jointplot(x='Glucose',y='BloodPressure',data=dataset,kind='reg')


# In[13]:


sns.pairplot(dataset,palette='coolwarm')


# In[17]:


sns.lmplot(x='DiabetesPedigreeFunction',y='Age',size=2,aspect=4,data=dataset)


# In[18]:


sns.lmplot(x='BMI',y='Age',size=2,aspect=4,data=dataset)


# In[34]:


X = dataset[feature_colu]
print(X)
y = dataset['Outcome']
print(y)


# In[35]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[36]:


print("X_train value \n ", X_train)
print("X_test value  \n ", X_test)
print("y_train value \n ",y_train)
print("y_test value \n ",y_test)


# In[37]:


from sklearn.linear_model import LogisticRegression
# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)
print(y_pred)


# In[38]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[39]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[40]:


y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

