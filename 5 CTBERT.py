from transformers import AutoConfig, AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
# import tensorflow_datasets as tfds
import pandas as pd

import numpy as np
#Comparison Time
from transformers import pipeline

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,precision_score,recall_score,f1_score

from sklearn.ensemble import VotingClassifier

# Load X_train, X_test, y_train, and y_test from the saved files
X_train = np.load('./Test_train for CTBERT/X_train_string.npy')
X_test = np.load('./Test_train for CTBERT/X_test_string.npy')
y_train = np.load('./Test_train for CTBERT/y_train.npy')
y_test = np.load('./Test_train for CTBERT/y_test.npy')


# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert')


#############################################
# Convert the text data to input features
train_input = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=128, return_tensors='tf')
test_input = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=128, return_tensors='tf')
train_input_dict = {key: value.numpy() for key, value in train_input.items()}
test_input_dict = {key: value.numpy() for key, value in test_input.items()}

# Configure the model and fine-tuning parameters
num_labels = 2  # Binary classification
learning_rate = 2e-5  # Define the desired learning rate here

config = AutoConfig.from_pretrained('digitalepidemiologylab/covid-twitter-bert', num_labels=num_labels)
model = TFAutoModelForSequenceClassification.from_pretrained('digitalepidemiologylab/covid-twitter-bert', config=config)

# Optimizer and loss
optimizer = Adam(learning_rate=learning_rate)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

# Compile and train the model
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# THIS TOOK 6 HOURS!!!
# WHYYYY
model.fit(train_input_dict, y_train, validation_data=(test_input_dict, y_test), epochs=3, batch_size=32)


# Save the fine-tuned model
model.save_pretrained('fine-tuned-ctbert')
# model.save('fine-tuned-ctbert.h5')
model.save('fine-tuned-ctbert', save_format='tf')
# model.save_weights('fine-tuned-ctbert.h5')

model_path = "fine-tuned-ctbert"
# Load the saved model
loaded_model = tf.keras.models.load_model(model_path)

# # Make predictions on the test set
# test_predictions = model.predict(test_dataset)
# predicted_labels = tf.argmax(test_predictions.logits, axis=1)
# Make predictions
predictions = loaded_model.predict(test_input_dict)
# predicted_labels = np.argmax(predictions.logits, axis=1)
predicted_labels = np.argmax(predictions['logits'], axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, predicted_labels)
print(f"Accuracy: {accuracy}")
ctbrt_confusion = confusion_matrix(y_test, predicted_labels)

print("CTBERT Confusion Matrix:")
print(ctbrt_confusion)

# Compute evaluation metrics
accuracy = accuracy_score(y_test, predicted_labels)
precision = precision_score(y_test, predicted_labels)
recall = recall_score(y_test, predicted_labels)
f1 = f1_score(y_test, predicted_labels)

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

##########################################################################################

# Load the pre-trained CT-BERT model

# # Save the fine-tuned model
# model.save_pretrained('ctbert-generic')
# # model.save('fine-tuned-ctbert.h5')
# model.save('ctbert-generic', save_format='tf')
# # model.save_weights('fine-tuned-ctbert.h5')

# # Load the saved model
# model_path = "ctbert-generic"
# loaded_model = tf.keras.models.load_model(model_path)

# Configure the model and fine-tuning parameters
num_labels = 2  # Binary classification
learning_rate = 2e-5  # Define the desired learning rate 

config = AutoConfig.from_pretrained('digitalepidemiologylab/covid-twitter-bert', num_labels=num_labels)
# Make predictions on the test data
# BE WARNED THIS TOOK HALF AN HOURRR
loaded_model = TFAutoModelForSequenceClassification.from_pretrained('digitalepidemiologylab/covid-twitter-bert', config=config)
predictions1 = loaded_model.predict(test_input_dict)

loaded_model = TFAutoModelForSequenceClassification.from_pretrained('digitalepidemiologylab/covid-twitter-bert', config=config)
predictions2 = loaded_model.predict(test_input_dict)

loaded_model = TFAutoModelForSequenceClassification.from_pretrained('digitalepidemiologylab/covid-twitter-bert', config=config)
predictions3 = loaded_model.predict(test_input_dict)

ensemble_logits = (predictions1['logits'] + predictions2['logits'] + predictions3['logits'] ) / 3
# Apply softmax to convert logits to probabilities
# Soft voting
ensemble_probs = tf.nn.softmax(ensemble_logits, axis=1)

# Get the predicted labels (index of the maximum probability)
predicted_labels = tf.argmax(ensemble_probs, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, predicted_labels)
print(f"Accuracy: {accuracy}")

# Compute evaluation metrics
precision = precision_score(y_test, predicted_labels)
recall = recall_score(y_test, predicted_labels)
f1 = f1_score(y_test, predicted_labels)

# Print the evaluation metrics
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

confusion = confusion_matrix(y_test, predicted_labels)
print("Confusion Matrix:")
print(confusion)
