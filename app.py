import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Title
st.title("🌸 Iris Flower Prediction App")

# Add image
st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", 
         caption="Iris Flower", use_container_width=True)

st.write("Enter the flower measurements to predict the species")

# Sidebar inputs
st.sidebar.header("Input Features")

def user_input():
    SepalLength = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.4)
    SepalWidth = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.4)
    PetalLength = st.sidebar.slider("Petal Length", 1.0, 7.0, 1.3)
    PetalWidth = st.sidebar.slider("Petal Width", 0.1, 2.5, 0.2)

    data = {
        'SepalLengthCm': SepalLength,
        'SepalWidthCm': SepalWidth,
        'PetalLengthCm': PetalLength,
        'PetalWidthCm': PetalWidth
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input()

# Show user input
st.subheader("User Input:")
st.write(input_df)

# Load dataset (you can also use your CSV)
df = pd.read_csv("Iris-checkpoint.csv")

X = df.drop("Species", axis=1)
y = df["Species"]

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Output
st.subheader("Prediction:")
st.success(prediction[0])

st.subheader("Prediction Probability:")
st.write(prediction_proba)