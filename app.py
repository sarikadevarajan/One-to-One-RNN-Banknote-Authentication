# One-to-One RNN using Banknote Authentication

# Dataset : BankNote_Authentication.csv

# --------------------------------------------
# Import Libraries
# --------------------------------------------

import os
import pickle

import pandas as pd
import numpy as np

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN

# --------------------------------------------
# Configuration
# --------------------------------------------

MODEL = "banknote_model.keras"

SCALER = "scaler.pkl"

# --------------------------------------------
# Train Model
# --------------------------------------------

def train_model():

    print("Loading Dataset...")

    df = pd.read_csv("C:\\Sarika\\Programming\\RNN\\onetoone\\BankNote_Authentication.csv")

    print(df.head())

    print(df.info())

    # ----------------------------------------
    # Features and Labels
    # ----------------------------------------

    X = df.iloc[:, :-1]

    y = df.iloc[:, -1]

    print("X Shape :", X.shape)

    print("Y Shape :", y.shape)

    # ----------------------------------------
    # Feature Scaling
    # ----------------------------------------

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    with open(SCALER, "wb") as f:

        pickle.dump(scaler, f)

    # ----------------------------------------
    # One-to-One RNN
    # ----------------------------------------

    X = X.reshape(

        X.shape[0],

        1,

        X.shape[1]

    )

    print("New Shape :", X.shape)

    # ----------------------------------------
    # Train Test Split
    # ----------------------------------------

    x_train, x_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42

    )
    
    
    
    # ----------------------------------------
    # Build Model
    # ----------------------------------------

    model = Sequential()

    # SimpleRNN Layer

    model.add(

        SimpleRNN(

            32,

            activation="relu",

            input_shape=(1, X.shape[2])

        )

    )

    # Output Layer

    model.add(

        Dense(

            1,

            activation="sigmoid"

        )

    )

    model.summary()

    # ----------------------------------------
    # Compile
    # ----------------------------------------

    model.compile(

        optimizer="adam",

        loss="binary_crossentropy",

        metrics=["accuracy"]

    )

    # ----------------------------------------
    # Train
    # ----------------------------------------

    history = model.fit(

        x_train,

        y_train,

        validation_split=0.2,

        epochs=30,

        batch_size=32

    )

    # ----------------------------------------
    # Save Model
    # ----------------------------------------

    model.save(MODEL)

    print("Model Saved Successfully")

    # ----------------------------------------
    # Evaluate
    # ----------------------------------------

    loss, accuracy = model.evaluate(

        x_test,

        y_test,

        verbose=1

    )

    print("\nAccuracy :", accuracy)

    # ----------------------------------------
    # Prediction
    # ----------------------------------------

    y_pred = model.predict(

        x_test,

        verbose=0

    )

    y_pred = (y_pred > 0.5).astype(int)

    # ----------------------------------------
    # Confusion Matrix
    # ----------------------------------------

    print("\nConfusion Matrix")

    print(

        confusion_matrix(

            y_test,

            y_pred

        )

    )

    # ----------------------------------------
    # Classification Report
    # ----------------------------------------

    print("\nClassification Report")

    print(

        classification_report(

            y_test,

            y_pred

        )

    )


# ----------------------------------------
# Train Model
# ----------------------------------------

if not os.path.exists(MODEL):

    train_model()
    
    
    
    
# ----------------------------------------
# Predict Banknote
# ----------------------------------------

def predict_banknote(variance, skewness, curtosis, entropy):

    # Load Model

    model = load_model(MODEL)

    # Load Scaler

    with open(SCALER, "rb") as f:

        scaler = pickle.load(f)

    # Create Input

    data = np.array([[

        variance,

        skewness,

        curtosis,

        entropy

    ]])

    # Scale

    data = scaler.transform(data)

    # Reshape for One-to-One RNN

    data = data.reshape(

        1,

        1,

        4

    )

    # Predict

    probability = model.predict(

        data,

        verbose=0

    )[0][0]

    if probability >= 0.5:

        return "Fake", probability

    else:

        return "Authentic", 1 - probability
    
    
    
    
    
# ----------------------------------------
# Streamlit UI
# ----------------------------------------

st.set_page_config(

    page_title="Banknote Authentication",

    page_icon="💵",

    layout="centered"

)

st.title("💵 Banknote Authentication")

st.write("### One-to-One RNN using SimpleRNN")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:

    variance = st.number_input(

        "Variance",

        value=0.0

    )

    skewness = st.number_input(

        "Skewness",

        value=0.0

    )

with col2:

    curtosis = st.number_input(

        "Curtosis",

        value=0.0

    )

    entropy = st.number_input(

        "Entropy",

        value=0.0

    )

st.markdown("")

if st.button(

    "🔍 Predict",

    use_container_width=True

):

    prediction, confidence = predict_banknote(

        variance,

        skewness,

        curtosis,

        entropy

    )

    if prediction == "Authentic":

        st.success("✅ Authentic Banknote")

    else:

        st.error("❌ Fake Banknote")

    st.metric(

        "Confidence",

        f"{confidence*100:.2f}%"

    )

st.markdown("---")

st.caption(

    "One-to-One RNN • TensorFlow • Streamlit"

)