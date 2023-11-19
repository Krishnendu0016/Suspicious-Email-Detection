# Importing the Dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import streamlit as st

# Data Collection & Pre-Processing
# loading the data from csv file to a pandas Dataframe
raw_mail_data = pd.read_csv('mail_data.csv')
raw_mail_data2 = pd.read_csv('spam_ham_dataset.csv')    

# replace the null values with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

# label spam mail as 0; ham mail as 1;
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# separating the data as texts and label
X1 = mail_data['Message']
Y1 = mail_data['Category']

X2 = raw_mail_data2['text']
Y2 = raw_mail_data2['label_num']

X = pd.concat([X1, X2], axis=0, ignore_index=True)
Y = pd.concat([Y1, Y2], axis=0, ignore_index=True)

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Transform the text data to feature vectors
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Convert Y_train and Y_test values as integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Model Selection and Training
model = RandomForestClassifier() 
model.fit(X_train_features, Y_train)

# Model Evaluation
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on training data:', accuracy_on_training_data)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
precision = precision_score(Y_test, prediction_on_test_data)
recall = recall_score(Y_test, prediction_on_test_data)
f1 = f1_score(Y_test, prediction_on_test_data)
roc_auc = roc_auc_score(Y_test, prediction_on_test_data)

print('Accuracy on test data:', accuracy_on_test_data)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
print('ROC AUC Score:', roc_auc)

# Streamlit App
st.title("Suspicious Email Detection")
input_mail = st.text_area("Enter the mail text:")
if st.button("Predict"):
    st.subheader(':blue[Result: ]')
    input_data_features = feature_extraction.transform([input_mail])
    prediction = model.predict(input_data_features)
    if prediction[0] == 1:
        st.subheader('This is not a suspicious email.')
    else:
        st.subheader('This is a suspicious spam email.')
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            text-align: center;
            background-color: #f1f1f1; 
            padding: 10px;
            font-size: 12px;
            color: black; 
        }
    </style>
    <div class="footer">
        <p>Copyright © 2023. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)