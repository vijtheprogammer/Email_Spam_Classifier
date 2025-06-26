# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Load the dataset
df = pd.read_csv('mail_data (1).csv')

# Replace null values with NaN (just to clean the data a bit)
data = df.where((pd.notnull(df)))

# Convert labels: 'spam' -> 0, 'ham' -> 1
data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1

# Separate the messages and labels
X = data['Message']
Y = data['Category']

# Split the data into training and testing sets (80/20 split)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Convert text data into numerical features using TF-IDF
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Make sure labels are integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Plot label distribution in training data
plt.figure(figsize=(10, 5))
sns.countplot(x=Y_train)
plt.title('Distribution of Target Labels in Training Set')
plt.xlabel('Labels (0 = Not Spam, 1 = Spam)')
plt.ylabel('Count')
plt.show()

# Plot label distribution in test data
plt.figure(figsize=(10, 5))
sns.countplot(x=Y_test)
plt.title('Distribution of Target Labels in Test Set')
plt.xlabel('Labels (0 = Not Spam, 1 = Spam)')
plt.ylabel('Count')
plt.show()

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Check model accuracy on training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data) 
print('Accuracy on training data:', accuracy_on_training_data)

# Check model accuracy on test data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data:', accuracy_on_test_data)

# Train the model again (redundant, but doesn't hurt)
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Get the feature names and coefficients from the model
feature_names = feature_extraction.get_feature_names_out()
coefficients = model.coef_.flatten()

# Get the top 20 positive and negative words (most important features)
top_positive_coefficients = np.argsort(coefficients)[-20:]
top_negative_coefficients = np.argsort(coefficients)[:20]
top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

# Visualize important features for classification
plt.figure(figsize=(15, 5))
colors = ['red' if c < 0 else 'blue' for c in coefficients[top_coefficients]]
plt.bar(np.array(feature_names)[top_coefficients], coefficients[top_coefficients], color=colors)
plt.title('Top 20 Important Features for Spam Classification')
plt.xticks(rotation=90)
plt.show()

# Make predictions on the test set
Y_pred = model.predict(X_test_features)

# Create a confusion matrix
cm = confusion_matrix(Y_test, Y_pred, labels=[0, 1])

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Spam', 'Spam'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Get predicted probabilities for ROC curve
Y_pred_proba = model.predict_proba(X_test_features)[:, 1]

# Calculate ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')  # Random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Create a folder to save the model if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the trained model to a file
with open('models/spam_classifier_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the feature extractor (TF-IDF) to a file
with open('models/feature_extractor.pkl', 'wb') as extractor_file:
    pickle.dump(feature_extraction, extractor_file)

# You can test the model with custom input here
input_mail = [""]  # Add a test message inside the quotes

# Transform input text using the saved feature extractor
input_mail_features = feature_extraction.transform(input_mail)

# Predict using the trained model
prediction = model.predict(input_mail_features)

# Print the result
if(prediction[0] == 0):
    print('Not Spam!')
else:
    print('Spam!')
