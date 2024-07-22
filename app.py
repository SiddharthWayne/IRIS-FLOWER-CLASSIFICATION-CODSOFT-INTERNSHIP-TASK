import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\siddh\OneDrive\Desktop\codsoft intern\IRIS\IRIS.csv")
    return df

# Preprocess the data
def preprocess_data(df):
    X = df.drop('species', axis=1)
    y = df['species']
    return X, y

# Train the model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy, classification_report(y_test, y_pred)

# Load and preprocess data
df = load_data()
X, y = preprocess_data(df)

# Train model
model, scaler, accuracy, report = train_model(X, y)

# Streamlit app
st.title("Iris Flower Classification")

# Sidebar for user input
st.sidebar.header("Flower Measurements")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

# Make prediction
if st.sidebar.button("Predict Species"):
    user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)
    probability = model.predict_proba(user_input_scaled)
    
    st.header("Prediction Result")
    st.success(f"The predicted Iris species is: {prediction[0]}")
    st.write("Prediction Probabilities:")
    st.write(pd.DataFrame(probability, columns=model.classes_))

# Display model performance
st.header("Model Performance")
st.write(f"Accuracy: {accuracy:.2f}")
st.text("Classification Report:")
st.text(report)

# Data Visualization
st.header("Data Visualization")

# Pairplot
st.subheader("Pairplot of Iris Features")
fig = sns.pairplot(df, hue='species')
st.pyplot(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
corr = df.drop('species', axis=1).corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Feature Importance
st.subheader("Feature Importance")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots()
sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
ax.set_title("Feature Importance in Iris Classification")
st.pyplot(fig)

# Show raw data
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(df)