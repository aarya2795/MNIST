
# coding: utf-8

# <h3> 2(a) Loads the necessary libraries to run (e.g., scikit-learn) </h3>

# In[1]:


import sklearn         
import statistics
import numpy as np
from sklearn.tree import DecisionTreeClassifier  # Importing Decision Tree Classifier
# For calculation of accuracy, importing scikit-learn metrics module
from sklearn import metrics    
# Importing module to get train/test indices for splitting data in train/test sets 
from sklearn.model_selection import KFold 
# Importing matrix module to get accuracy score on testing model performance
from sklearn.metrics import accuracy_score  


# <h3> 2(b) Loads the dataset “digits” (see below for description) from the scikit-learn example datasets package </h3>

# In[2]:


# Importing 'digits' dataset from scikit-learn e.g. dataset package for classification 
from sklearn.datasets import load_digits 
# Loading dataset to create a bunch lettting Python dictionary to be used like an object
digits_dataset = load_digits()                    


# In[3]:


features = digits_dataset.data          # Creating the feature matrix

class_labels = digits_dataset.target    # Create target vector having all class labels


# <h3>2(c) Instantiates a sklearn.tree.DecisionTreeClassifier</h3>

# In[4]:


DT_classifier = DecisionTreeClassifier() # Creating DecisionTreeClassifer object


# <h3>2(d) Splits dataset in training & testing sets using sklearn.model_selection.KFold cross validation (K = 5)</h3>

# In[5]:


K_Folds = KFold(n_splits=5)     # Using KFold, split dataset in 'k' consecutive folds


# <h3> 2(e) Trains and tests the performance of the classifier in identifying the digits. </h3>
# <h3> 2(f).1 Computes the accuracy for each of the K folds of cross validation </h3>

# In[6]:


# Creating list to store accuracy of each fold to take an average of all values
accuracy_list = [] 

for training_index, testing_index in K_Folds.split(features):
    # Using 'split' method to generate indices for splitting data in training/testing 
    # and loop over the indices.
    # 
    # Creating 'feature_train and 'feature_test' ndarrays w.r.t training_index 
    # and testing_index respectively.
    # 
    # Creating 'label_train and 'label_test' ndarrays w.r.t training_index 
    # and testing_index respectively. 
    # 
    # Trainig the decision tree classifier.  
    # 
    # Predicting response for the test dataset.
    # 
    # Calculating model accuracy for the fold.
    # 
    # Add the accuracy value obtained to the list used to store all accuracies.
    #
    feature_train, feature_test = features[training_index], features[testing_index]  
    label_train, label_test = class_labels[training_index], class_labels[testing_index]
    trained_model = DT_classifier.fit(feature_train, label_train)        
    label_prediction = trained_model.predict(feature_test)               
    accuracy = (accuracy_score(label_test, label_prediction) * 100)      
    accuracy_list.append(accuracy)


# <h3>2(f).2 Outputs the accuracy for each of the K folds of cross validation</h3>

# In[7]:


fold_counter = 1            # Creating a variable to keep track of the fold number
for accuracy in accuracy_list:
    print("Accuracy for fold", fold_counter, "=", accuracy)
    fold_counter += 1


# <h3>2(g) Computes and outputs the average accuracy across the K folds</h3>

# In[8]:


print("Average accuracy across the K folds =", statistics.mean(accuracy_list))


# <h3>(EXTRA) Testing the performance of the classifier in identifying the digits<br> with given feature ndarray</h3>

# In[9]:


# Passing the feature array of digit '5' as input for prediction
model_identified_digit = trained_model.predict([features[5]])                            
print("Model identified the given digit-input as digit =", model_identified_digit.item())


# <h3> Tests the performance of the classifier in identifying the digits (With slight modifications to the feature array of digit '9')</h3>
# <h6>Original array is:-</h6>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
#                     array([ 0.,  0., 11., 12.,  0.,  0.,  0.,  0.,  0.,  2., 16., 16., 16.,<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
#                            13.,  0.,  0.,  0.,  3., 16., 12., 10., 14.,  0.,  0.,  0.,  1.,                          <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
#                            16.,  1., 12., 15.,  0.,  0.,  0.,  0., 13., 16.,  9., 15.,  2.,<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
#                             0.,  0.,  0.,  0.,  3.,  0.,  9., 11.,  0.,  0.,  0.,  0.,  0.,<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
#                             9., 15.,  4.,  0.,  0.,  0.,  9., 12., 13.,  3.,  0.,  0.])
#                             
# <h6>We modify it to the following to test performance:-</h6><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
#                     array([ 0.,  0., 10., 12.,  0.,  0.,  0.,  0.,  0.,  1., 16., 15., 16.,<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
#                            13.,  0.,  0.,  0.,  3., 15., 12., 12., 14.,  0.,  0.,  0.,  1.,<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
#                            16.,  1., 12., 15.,  0.,  0.,  0.,  0., 13., 16.,  10., 15.,  2.,<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
#                             0.,  0.,  0.,  0.,  3.,  0.,  9., 11.,  0.,  0.,  0.,  0.,  0.,<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
#                             10., 15.,  5.,  1.,  0.,  0.,  10., 12., 13.,  1.,  0.,  0.])
#                             

# In[10]:


'''2(e) Tests the performance of the classifier in identifying the digits.
        (With slight modifications to the feature array of digit '9')

Original array is:-

   array([ 0., 0., 11., 12., 0., 0., 0., 0., 0., 2., 16., 16., 16.,
          13., 0., 0., 0., 3., 16., 12., 10., 14., 0., 0., 0., 1.,
          16., 1., 12., 15., 0., 0., 0., 0., 13., 16., 9., 15., 2.,
           0., 0., 0., 0., 3., 0., 9., 11., 0., 0., 0., 0., 0.,
           9., 15., 4., 0., 0., 0., 9., 12., 13., 3., 0., 0.])
   
   
We modify it to the following to test performance:-

   array([ 0., 0., 10., 12., 0., 0., 0., 0., 0., 1., 16., 15., 16.,
         13., 0., 0., 0., 3., 15., 12., 12., 14., 0., 0., 0., 1.,
         16., 1., 12., 15., 0., 0., 0., 0., 13., 16., 10., 15., 2.,
          0., 0., 0., 0., 3., 0., 9., 11., 0., 0., 0., 0., 0.,
         10., 15., 5., 1., 0., 0., 10., 12., 13., 1., 0., 0.])
'''

modified_array = np.array([ 0.,  0., 11., 12.,  0.,  0.,  0.,  0.,  0.,  1., 16., 15., 16.,
                            13.,  0.,  0.,  0.,  2., 16., 12., 11., 14.,  0.,  0.,  0.,  1.,
                            16.,  1., 11., 15.,  0.,  0.,  0.,  0., 13., 16.,  8., 15.,  1.,
                             0.,  0.,  0.,  0.,  3.,  0.,  9., 10.,  0.,  0.,  0.,  0.,  0.,
                             8., 15.,  5.,  1.,  0.,  0.,  10., 12., 13.,  2.,  0.,  0.])


# In[11]:


# Passing the modified feature array as input for prediction
model_identified_digit = trained_model.predict([modified_array])                
print("Model identified given digit-input as digit:", model_identified_digit.item())

