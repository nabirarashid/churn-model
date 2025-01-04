from tensorflow.keras.models import load_model

# Load the existing .h5 model
model = load_model("churn_model.h5")

# Save the model in the native Keras format
model.save("churn_model.keras")

print("Model has been converted and saved as churn_model.keras")