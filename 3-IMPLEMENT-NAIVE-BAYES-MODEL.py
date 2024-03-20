import pandas as pd

msg=pd.read_csv('naivetext.csv',names=['message','label'])

print('The dimensions of the dataset',msg.shape)

msg['labelnum']=msg.label.map({'pos':1,'neg':0})

X=msg.message

y=msg.labelnum

print(X)

print(y)



#splitting the dataset into train and test data

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(X,y)

print ('\n The total number of Training Data :',ytrain.shape)

print ('\n The total number of Test Data :',ytest.shape)



#output of count vectoriser is a sparse matrix

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()

xtrain_dtm = count_vect.fit_transform(xtrain)

xtest_dtm=count_vect.transform(xtest)

print('\n The words or Tokens in the text documents \n')

print(count_vect.get_feature_names())

df=pd.DataFrame(xtrain_dtm.toarray(),columns=count_vect.get_feature_names())



# Training Naive Bayes (NB) classifier on training data.

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(xtrain_dtm,ytrain)

predicted = clf.predict(xtest_dtm)



#printing accuracy, Confusion matrix, Precision and Recall

from sklearn import metrics

print('\n Accuracy of the classifer is',

      metrics.accuracy_score(ytest,predicted))

print('\n Confusion matrix')

print(metrics.confusion_matrix(ytest,predicted)) 

print('\n The value of Precision' ,

 metrics.precision_score(ytest,predicted)) 

print('\n The value of Recall') ,

metrics.recall_score(ytest,predicted)

# OUTPUT
# The dimensions of the dataset (18, 2)
# 0 I love this sandwich
# 1 This is an amazing place
# 2 I feel very good about these beers
# 3 This is my best work
# 4 What an awesome view
# 5 I do not like this restaurant
# 6 I am tired of this stuff
# 7 I can't deal with this
# 8 He is my sworn enemy
# 9 My boss is horrible
# 10 This is an awesome place
# 11 I do not like the taste of this juice
# 12 I love to dance
# 13 I am sick and tired of this place
# 14 What a great holiday
# 15 That is a bad locality to stay
# 16 We will have good fun tomorrow
# 17 I went to my enemy's house today

# Name: message, dtype: object
# 0 1
# 1 1
# 2 1
# 3 1
# 4 1
# 5 0
# 6 0
# 7 0
# 8 0
# 9 0
# 10 1
# 11 0
# 12 1
# 13 0
# 14 1
# 15 0
# 16 1
# 17 0
# Name: labelnum, dtype: int64
# The total number of Training Data: (13,)
# The total number of Test Data: (5,)
# The words or Tokens in the text documents
# ['about', 'am', 'amazing', 'an', 'and', 'awesome', 'beers', 'best', 'can', 'deal', 'do', 'enemy', 'feel',
# 'fun', 'good', 'great', 'have', 'he', 'holiday', 'house', 'is', 'like', 'love', 'my', 'not', 'of', 'place',
# 'restaurant', 'sandwich', 'sick', 'sworn', 'these', 'this', 'tired', 'to', 'today', 'tomorrow', 'very',
# 'view', 'we', 'went', 'what', 'will', 'with', 'work']
# Accuracy of the classifier is 0.8
# Confusion matrix
# [[2 1]
# [0 2]]
# The value of Precision 0.6666666666666666
# The value of Recall 1.0
