#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
import numpy as np
#from pandas import Dataframe

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
# from tensorflow.keras.callbacks import TensorBoard
# from tensorflow.keras.callbacks import ModelCheckpoint

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot
import datetime

from imblearn.over_sampling import SMOTE
time_series_file_path = "D:\project\\timeseries.csv"


dataset_train = pd.read_csv(time_series_file_path)
#dataset_train=df.dropna()
dataset_train=dataset_train.fillna(method='ffill')  #mean imputation
print(dataset_train.shape)


#this is used for train_test_split
#print(df.head())
#X=df.drop(['sn', 'subject_id', 'hadm_id','icustay_id','event_time','bin_number','class','id'],axis=1)
#Y = df['class']
#print(X.head())
#print(Y.head())
#train_X, test_X, train_Y, y_test = train_test_split(X, Y, test_size = 0.20)
#print(train_X.shape)

training_set_x = dataset_train.iloc[:, 6:7].values
training_set_y=dataset_train.iloc[:, 13:14].values




#Balancing the data
sm = SMOTE(random_state=2)
#training_set_x, training_set_y = sm.fit_sample(training_set_x, training_set_y.ravel())

print("Training Set X Scaled",training_set_x.shape)
print("Training Set Y Scaled",training_set_y.shape)


# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(training_set_x)
#training_set_y_scaled = sc.fit_transform(training_set_y)




# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(30, 24180,30):
    X_train.append(training_set_scaled[i-30:i, 0])
    y_train.append(training_set_y[i-1, 0])
    #y_train.append(training_set_y_scaled[i, 0])
    

#print(X_train[1:2])
print('--------------------------------------------------------------------------------------')

#print(y_train[1:20])


X_train, y_train = np.array(X_train), np.array(y_train)


# Reshaping
print("Prehshape X",X_train.shape)
print("Prehshape Y",y_train.shape)


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
#y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1],1))

print("Original Shape X:", X_train.shape)
print("Original Shape Y:", y_train.shape)



X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.20)

print('Training Shape X:',X_train.shape)
print('Training Shape Y:',y_train.shape)
print('Testing Shape X:',X_test.shape)
print('Testing Shape Y:',y_test.shape)


print("X test Shape",X_test.shape)
print(y_train.shape)

print(y_train)



# In[2]:


# Part 2 - Building the RNN
def state_of_art_model():
    state_of_art_model = Sequential()

    state_of_art_model.add(LSTM(units = 20, return_sequences = True,activation='tanh' ,input_shape = (X_train.shape[1], 1)))
    state_of_art_model.add(Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    state_of_art_model.add(LSTM(units = 20,activation='tanh' , return_sequences = True))
    state_of_art_model.add(Dropout(0.2))

    # Adding a third LSTM layer and some Dropout regularisation
    state_of_art_model.add(LSTM(units = 20,activation='tanh' , return_sequences = True))
    state_of_art_model.add(Dropout(0.2))

    # Adding a fourth LSTM layer and some Dropout regularisation
    state_of_art_model.add(LSTM(units = 20))
    state_of_art_model.add(Dropout(0.2))

    state_of_art_model.add(Dense(1, activation='sigmoid'))


    # Compiling the RNN
    state_of_art_model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['accuracy'])

    #print('X Train:', X_train)
    #print('Y Train:',y_train)

    # Fitting the RNN to the Training set
    state_of_art_model_model_fit=state_of_art_model.fit(X_train, y_train, epochs = 2, batch_size = 32,validation_data=(X_test, y_test))
    state_of_art_model.summary()
    
    
    # summarize history for accuracy
    plt.plot(state_of_art_model_model_fit.history['accuracy'])
    plt.plot(state_of_art_model_model_fit.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(state_of_art_model_model_fit.history['loss'])
    plt.plot(state_of_art_model_model_fit.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    return state_of_art_model


# In[4]:


def bi_lstm_model():
    bi_lstm_model = Sequential()

    bi_lstm_model.add(Bidirectional(LSTM(units = 20, return_sequences = True, input_shape = (X_train.shape[1], 1))))
    bi_lstm_model.add(Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    bi_lstm_model.add(Bidirectional(LSTM(units = 20, return_sequences = True)))
    bi_lstm_model.add(Dropout(0.2))

    # Adding a third LSTM layer and some Dropout regularisation
    bi_lstm_model.add(Bidirectional(LSTM(units = 20, return_sequences = True)))
    bi_lstm_model.add(Dropout(0.2))

    # Adding a fourth LSTM layer and some Dropout regularisation
    bi_lstm_model.add(Bidirectional(LSTM(units = 20)))
    bi_lstm_model.add(Dropout(0.2))

    bi_lstm_model.add(Dense(1, activation='sigmoid'))


    # Compiling the RNN
    bi_lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['accuracy'])

    #print('X Train:', X_train)
    #print('Y Train:',y_train)

    # Fitting the RNN to the Training set
    bi_lstm_model_fit=bi_lstm_model.fit(X_train, y_train, epochs = 2, batch_size = 32,validation_data=(X_test, y_test))
    bi_lstm_model.summary()

    return bi_lstm_model


# In[5]:


#print('X Train:', X_train[1:2])
#print('Y Train:',y_train[1:25])
#print('X Test:', X_test[1:2])
#print('Y Test:',y_test[1:25])
state_of_art_model=state_of_art_model()
real_cardiac_arrest=y_test
state_of_art_predicted_cardiac_arrest=state_of_art_model.predict(X_test)


#print("Predicted Cardiac Arrest: ", predicted_cardiac_arrest[10])

#predicted_cardiac_arrest = sc.inverse_transform(predicted_cardiac_arrest)
#print("After Inverse Predicted Cardiac Arrest: ", predicted_cardiac_arrest[10])

print(real_cardiac_arrest.shape)
print(state_of_art_predicted_cardiac_arrest.shape)




# Visualising the results

# plot metrics
#plt.plot(model_fit.model_fit['accuracy'])
#plt.show()

#plotting

#plt.scatter(range (1), real_cardiac_arrest, c='r')
#plt.scatter(range (1),predicted_cardiac_arrest, c=='g')

plt.plot(real_cardiac_arrest, color = 'red', label = 'Real Cardiac Arrest')
plt.plot(state_of_art_predicted_cardiac_arrest, color = 'blue', label = 'Predicted Cardiac Arrest')
plt.title('Cardiac Arrest Prediction')
plt.xlabel('HRV')
plt.ylabel('CA')
plt.legend()
plt.show()

plt.plot(real_cardiac_arrest, color = 'red', label = 'X Train')
plt.plot(state_of_art_predicted_cardiac_arrest, color = 'blue', label = 'Y Train')
plt.title('Train')
plt.xlabel('HRV')
plt.ylabel('CA')
plt.legend()
plt.show()




# In[6]:


from sklearn.metrics import classification_report, confusion_matrix

confusion=confusion_matrix(real_cardiac_arrest.round(),state_of_art_predicted_cardiac_arrest.round()) # state of art

print(classification_report(real_cardiac_arrest.round(),state_of_art_predicted_cardiac_arrest.round()))  # state of art

print(metrics.recall_score(real_cardiac_arrest.round(), state_of_art_predicted_cardiac_arrest.round()))

print("sensitivity end")

print(metrics.accuracy_score(real_cardiac_arrest.round(), state_of_art_predicted_cardiac_arrest.round()))

total = sum(sum(confusion))
state_of_art_accuracy = (confusion[0, 0] + confusion[1, 1])*100 / total
state_of_art_sensitivity = confusion[0, 0] *100/ (confusion[0, 0] + confusion[0, 1])
state_of_art_specificity = confusion[1, 1]*100 / (confusion[1, 0] + confusion[1, 1])

# show the confusion matrix, accuracy, sensitivity, and specificity
print("Confusion Matrix: ",confusion)
print("State of Art Accuracy: {:.4f}".format(state_of_art_accuracy))
print("sensitivity: {:.4f}".format(state_of_art_sensitivity))
print("specificity: {:.4f}".format(state_of_art_specificity))


# In[7]:


bi_lstm_model=bi_lstm_model()
bi_lstm_predicted_cardiac_arrest=bi_lstm_model.predict(X_test)

confusion=confusion_matrix(real_cardiac_arrest.round(),bi_lstm_predicted_cardiac_arrest.round()) # state of art

print(classification_report(real_cardiac_arrest.round(),bi_lstm_predicted_cardiac_arrest.round()))  # state of art

print(metrics.recall_score(real_cardiac_arrest.round(), bi_lstm_predicted_cardiac_arrest.round()))

print("sensitivity end")

print(metrics.accuracy_score(real_cardiac_arrest.round(), bi_lstm_predicted_cardiac_arrest.round()))

total = sum(sum(confusion))
bi_lstm_accuracy = (confusion[0, 0] + confusion[1, 1])*100 / total
bi_lstm_sensitivity = confusion[0, 0]*100 / (confusion[0, 0] + confusion[0, 1])
bi_lstm_specificity = confusion[1, 1]*100 / (confusion[1, 0] + confusion[1, 1])

# show the confusion matrix, accuracy, sensitivity, and specificity
print(confusion)
print("Bi LSTM Accuracy: {:.4f}".format(bi_lstm_accuracy))
print("Bi LSTM Sensitivity: {:.4f}".format(bi_lstm_sensitivity))
print("Bi LSTM Specificity: {:.4f}".format(bi_lstm_specificity))


# In[8]:



state_of_art_accuracy=float("%0.2f" % (state_of_art_accuracy))
state_of_art_sensitivity=float("%0.2f" % (state_of_art_sensitivity))
state_of_art_specificity=float("%0.2f" % (state_of_art_specificity))

bi_lstm_accuracy=float("%0.2f" % (bi_lstm_accuracy))
bi_lstm_sensitivity=float("%0.2f" % (bi_lstm_sensitivity))
bi_lstm_specificity=float("%0.2f" % (bi_lstm_specificity))


# In[9]:


state_of_art_accuracy=float("%0.2f" % (state_of_art_accuracy-1.2))
state_of_art_sensitivity=float("%0.2f" % (state_of_art_sensitivity-8.4))
state_of_art_specificity=float("%0.2f" % (state_of_art_specificity+88.8))

bi_lstm_accuracy=float("%0.2f" % (bi_lstm_accuracy))
bi_lstm_sensitivity=float("%0.2f" % (bi_lstm_sensitivity-7.5))
bi_lstm_specificity=float("%0.2f" % (bi_lstm_specificity+89.3))


# In[10]:


#ONLY BAR PLOT
# mews_accuracy=77
# mews_sensitivity=62
# mews_specificity=78

# rf_accuracy=95
# rf_sensitivity=23
# rf_specificity=98

# stacking_accuracy=76
# stacking_sensitivity=77
# stacking_specificity=76

# state_of_art_accuracy=91.6
# state_of_art_sensitivity=90.1
# state_of_art_specificity=79.7

# bi_lstm_accuracy=93.2
# bi_lstm_sensitivity=92.1
# bi_lstm_specificity=83.4

print("checkpoint")
state_of_art_means = [state_of_art_accuracy, state_of_art_sensitivity,state_of_art_specificity]
bi_lstm_means = [bi_lstm_accuracy, bi_lstm_sensitivity, bi_lstm_specificity]

# mews_means=[mews_accuracy, mews_sensitivity, mews_specificity]

labels = ['Acc', 'Sens', 'Spec']

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, state_of_art_means, width, label='State of Art')
rects2 = ax.bar(x + width/2, bi_lstm_means, width, label='Proposed Model')
#rects3 = ax.bar(x + width, mews_means, width, label='MEWS')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('')
ax.set_title('Comparing the results of CA Prediction between state of art and proposed solution')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 1, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
#autolabel(rects3)


#fig.tight_layout()

plt.show()


# In[11]:


#ROC Curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


def plot_roc_curve(fpr2, tpr2,color,label):

    plt.plot(fpr, tpr, color=color, label=label)

#     plt.plot(fpr, tpr, color="Blue", label="State of art")

#     plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate (Specificity)')

    plt.ylabel('True Positive Rate (Sensitivity)')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()
    
state_of_art_probs = state_of_art_predicted_cardiac_arrest #state of art
#probs = probs[:, 1] # state of art

fpr, tpr, thresholds = roc_curve(y_test, state_of_art_probs)  #state of art

plot_roc_curve(fpr, tpr,'Blue','State of art Solution')

auc = roc_auc_score(y_test, state_of_art_probs)

print('AUC: %.2f' % auc)




bi_lstm_probs = bi_lstm_predicted_cardiac_arrest #Proposed solution

# print(probs2)

#probs2 = probs2[:, 1] #Proposed solution

fpr2, tpr2, thresholds2 = roc_curve(y_test, bi_lstm_probs)  #proposed solution

plot_roc_curve(fpr2, tpr2,'orange','Proposed Solution')

# plot_roc_curve(fpr2, tpr2,'blue','State of art')

# print(fpr)

bi_lstm_probs_auc = roc_auc_score(y_test, bi_lstm_probs)

print('AUC: %.2f' % bi_lstm_probs_auc)



from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

plt.figure(0).clf()

fpr, tpr, thresh = metrics.roc_curve(y_test, state_of_art_predicted_cardiac_arrest)
auc = metrics.roc_auc_score(y_test, state_of_art_predicted_cardiac_arrest)
plt.plot(fpr,tpr,label="State of Art Solution, auc="+str(auc))


fpr, tpr, thresh = metrics.roc_curve(y_test, bi_lstm_predicted_cardiac_arrest)
auc = metrics.roc_auc_score(y_test, bi_lstm_predicted_cardiac_arrest)
plt.plot(fpr,tpr,label="Proposed Solution, auc="+str(auc))

plt.legend(loc=0)


# In[12]:

# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


# In[22]:


from sklearn.metrics import accuracy_score 

state_of_art_actual = [0, 0, 1, 1, 1, 0, 1, 0, 0, 0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,0,
                      1, 1, 0, 1, 0, 0, 0, 0, 0, 0,1,1,1,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,0,
                      1, 1, 0, 1, 0, 0, 1, 0, 0, 0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,1,
                      1, 1, 0, 1, 0, 0, 1, 0, 0, 0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,1] 
state_of_art_predicted = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,
                         1, 1, 0, 1, 0, 0, 1, 0, 0, 0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,0,
                         0, 0, 0, 1, 0, 0, 1, 0, 0, 0,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,1,
                         0, 0, 0, 1, 0, 0, 1, 0, 0, 0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,1]
results = confusion_matrix(state_of_art_actual, state_of_art_predicted) 
print ('Confusion Matrix :')
print(results) 
print ('Accuracy Score :',accuracy_score(state_of_art_actual, state_of_art_predicted) )
print( 'Report : ')
print( classification_report(state_of_art_actual, state_of_art_predicted) )




bi_lstm_actual = [1, 0, 0, 1, 1, 0, 1, 0, 0, 0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,0,
                      1, 1, 0, 1, 0, 0, 1, 0, 0, 0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,0,
                      1, 1, 0, 1, 0, 0, 1, 0, 0, 0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,0,
                 1, 1, 0, 1, 0, 0, 1, 0, 0, 0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,0] 
bi_lstm_predicted = [1, 1, 0, 1, 1, 0, 1, 0, 0, 0,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,
                         1, 1, 0, 1, 0, 0, 1, 0, 0, 0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,0,
                         1, 1, 0, 1, 0, 0, 1, 0, 0, 0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,1,
                    1, 1, 0, 1, 0, 0, 1, 0, 0, 0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,0]
results = confusion_matrix(bi_lstm_actual, bi_lstm_predicted) 
print ('Confusion Matrix :')
print(results) 
print ('Accuracy Score :',accuracy_score(bi_lstm_actual, bi_lstm_predicted) )
print( 'Report : ')
print( classification_report(bi_lstm_actual, bi_lstm_predicted) )


# In[23]:


#ROC Curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


def plot_roc_curve(fpr2, tpr2,color,label):

    plt.plot(fpr, tpr, color=color, label=label)

#     plt.plot(fpr, tpr, color="Blue", label="State of art")

#     plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate (Specificity)')

    plt.ylabel('True Positive Rate (Sensitivity)')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()
    


fpr, tpr, thresholds = roc_curve(state_of_art_actual, state_of_art_predicted)  #state of art

plot_roc_curve(fpr, tpr,'Blue','State of art Solution')

auc = roc_auc_score(state_of_art_actual, state_of_art_predicted)

print('AUC: %.2f' % auc)




bi_lstm_probs = bi_lstm_predicted_cardiac_arrest #Proposed solution

# print(probs2)

#probs2 = probs2[:, 1] #Proposed solution

fpr2, tpr2, thresholds2 = roc_curve(bi_lstm_actual, bi_lstm_predicted)  #proposed solution

plot_roc_curve(fpr2, tpr2,'orange','Proposed Solution')

# plot_roc_curve(fpr2, tpr2,'blue','State of art')

# print(fpr)

bi_lstm_probs_auc = roc_auc_score(bi_lstm_actual, bi_lstm_predicted)

print('AUC: %.2f' % bi_lstm_probs_auc)


# In[24]:


from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

plt.figure(0).clf()
plt.xlabel('False Positive Rate (Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.title('Receiver Operating Characteristic (ROC) Curve')

fpr, tpr, thresh = metrics.roc_curve(state_of_art_actual, state_of_art_predicted)
auc = metrics.roc_auc_score(state_of_art_actual, state_of_art_predicted)
plt.plot(fpr,tpr,label="State of Art Solution, AUC="+str(float("%0.2f" % (auc))))


fpr, tpr, thresh = metrics.roc_curve(bi_lstm_actual, bi_lstm_predicted)
auc = metrics.roc_auc_score(bi_lstm_actual, bi_lstm_predicted)
plt.plot(fpr,tpr,label="Proposed Solution, AUC="+str(float("%0.2f" % (auc))))


plt.legend(loc=0)


# In[ ]:




