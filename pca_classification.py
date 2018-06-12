# APPLYING LDA

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import connect
import math

db=connect.db
cur=db.cursor()
# Importing the dataset

dataset=pd.read_sql_query("select * from ml_training_set where payment_status>=0",db)
X = dataset.iloc[:,[4,6,7,8,9,10,11,12,13,16]].values
y = dataset.iloc[:, 14].values

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

X_holder=X_test
# Feature Scaling

row_ids=dataset.iloc[:,0]

if len(row_ids)<1:
    print('No data to classify')
else:
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Applying PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components = None)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_
    
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 30, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    
    #my_list=[y_pred,y_test,X_test]
    
    y_all=[y_pred.reshape(-1,1),y_test.reshape(-1,1)]
    y_all2=np.array(y_all).reshape(320,2)
    
    y_allnew=np.concatenate((y_all2,X_holder),axis=1)

    print(pd.DataFrame(y_allnew,columns=['y_pred','y_test','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10']))
        
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print ('\n**********************  EXPLAINED VARIANCE ***************************')
    print (explained_variance)
    
    print ('\n************CUMULATIVE SUM OF THE EXPLAINED VARIANCE*****************')
    print (pca.explained_variance_ratio_.cumsum())
    
    #print('\nPCA SCALINGS \n')
    my_saclings=pca.components_
    #print (pd.DataFrame(my_saclings))"""
    
    print('**************************  CONFUSION METRICS ****************************\n')
    
    sum=0.0
    for i in range(4):
    	for j in range(4):
    		num=cm[i,j] 
    		sum+=num


    print(pd.DataFrame(cm,columns=['Not Paid','Paid','was bad Debt','was overdue']))
    diag=cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3]

    percent=(diag/sum)*100

    print ('\n**************************** ACCURACY')
    print (percent,'%')

    print ('\n***************************** CONTRIBUTING FACTORS ******************************')

    variables=['user_friends','amount_borrowed','guarantor','family','friends','work','education','tagged_photos','albums','referee']

    print ('\n')
    for i in range(10):
        if explained_variance[i]<0.15:
            break
        print ('************************** COMPONENT ',i+1,' VARIANCE EXPLAINED: ', str('%.2f'%(explained_variance[i]*100))+'%')
        for j in range(10):
            print (variables[j],'     ',my_saclings[j,i],' PERCENTAGE: ',str('%.2f'%abs((my_saclings[j,i])*100))+'%')
        print ('\n')








