import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

df = pd.read_csv("Churn.csv")

# converts categorical variable(s) into dummy/indicator variables
# when the algorithm expects numerical input

# drop churn and customer id columns from input features
x = pd.get_dummies(df.drop(["Churn", "Customer ID"], axis =1))

# target variable; returns 1 if Churn is "Yes" and 0 if "No"
y = df["Churn"].apply(lambda x: 1 if x =="Yes" else 0)

# split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# adding neural network layers to the model
model = Sequential()

# input layer has 32 neurons and uses ReLU activation function
# input_dim is the number of features in the input
model.add(Dense(units = 32, activation = "relu", input_dim = len(x_train.columns)))

# hidden layer has 64 neurons and uses ReLU activation function
model.add(Dense(units = 64, activation = "relu"))

# final output layer has one neuron - Yes or No to churning
model.add(Dense(units = 1, activation = "sigmoid"))

# optimizer helps to reduce the loss function
model.compile(optimizer = "sgd", loss = "binary_crossentropy", metrics = ["accuracy"])

# train the model; epochs is the number of iterations
model.fit(x_train, y_train, epochs = 200, batch_size = 32)

model.save("churn_model.h5")