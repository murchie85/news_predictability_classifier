#-----------------------------------------------------------------------------------------
#
# Program name : classifier.py
# Program Created Date: 16 Jan 2020
# Program Amanded Date: 
# Program Author: Adam McMurchie
#
# Program Description:  
#                       
#                       
#
# Input Files:     Data csv file, to try the program you can use sample_train.csv 
#                  but the results will be very poor as you need high volume data.
# Output Files:    Json files in data folder, output.csv full list in data folder
#
# Program Flow :  **to be updated** 
#
#
#-----------------------------------------------------------------------------------------

import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection, preprocessing, metrics
import numpy as np

# Read in CSV of news data
data = pd.read_csv('data.csv')
print('The count of news sources to train on are: ')
print(data['source'].value_counts())


#target = input('')


# Preprocessing
data = data.drop(columns=['Unnamed: 0'],axis=1)             # Remove dud index column
data = data.dropna()										# Remove NaN values
data['target'] = np.where(data['source']=='Reuters', 1, 0)  # Change for desired news outlet
data = data.drop(columns='source', axis=1) 					# removing leakage
data = data.drop(columns='author', axis=1) 					# removing leakage
data = data.drop(columns='url', axis=1)					    # removing leakage


# Shuffle Data
data = data.sample(frac=1)


print(data.head())


# Train/Test Split (test is the validation in this case: will split 80:10:10 in future)
valid_fraction=0.2
valid_rows = int(len(data) * valid_fraction)
train = data[:-valid_rows]
test = data[-valid_rows:]


# Initialize Count vectorizer
count_vectorizer = feature_extraction.text.CountVectorizer()


# Show an example 
example_train_vectors = count_vectorizer.fit_transform(train["title"][0:5])
print(example_train_vectors[0].todense().shape)
print('')
print(example_train_vectors[0].todense())


# Remove target from test as to prevent leakage
test.drop(['target'], axis=1)


# Set up Train and Test vectors on the headlines
train_vectors = count_vectorizer.fit_transform(train["title"])
test_vectors = count_vectorizer.transform(test["title"])


# Initialise classifier
# Moving forward will experiment with more advanced models (TFIDF, LSA, LSTM / RNNs) 
clf = linear_model.RidgeClassifier()

# Perform cross validation to get a feel for accuracy
scores = model_selection.cross_val_score(clf, train_vectors, train["target"], cv=3, scoring="f1")
print('The scores are: ' + str(scores))

# Fitting the model train vectors with train target
clf.fit(train_vectors, train["target"])

# Perform predictions on test data
test["prediction"] = clf.predict(test_vectors)

# Add a column to show if the clf got it right or not for that row
test['correct'] = np.where(test['target']==test['prediction'], 'correct', 'incorrect')

print('Overall Accuracy: ')
print(test['correct'].value_counts())


target_only = test[test.target == 1]
print('Prediction accuracy when only considering the target outlet: ')
print(target_only['correct'].value_counts())


while True:
    print('Do you want to save the prediction?')
    decision = input('Enter Y or N').upper()
    if decision == 'Y':
        print('Saving...')
        test.to_csv('predictions.csv')
        break
    elif decision =='N':
        print('Exiting')
        break