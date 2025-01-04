# filepath: /Users/user/Documents/Projects/churn-tn-model/predict.py
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

# Load the model
model = load_model("churn_model.keras")

# Load new data for prediction
df = pd.read_csv("Churn.csv")

# converts categorical variable(s) into dummy/indicator variables
# when the algorithm expects numerical input

x = pd.get_dummies(df.drop(["Churn", "Customer ID"], axis =1))
y = df["Churn"].apply(lambda x: 1 if x =="Yes" else 0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# Make a prediction
y_hat = predictions = model.predict(x_test)
y_hat = [0 if i < 0.5 else 1 for i in y_hat]

# Print predictions
print(y_hat, accuracy_score(y_test, y_hat)) # Accuracy: 0.77
print("Predictions have been made successfully") 