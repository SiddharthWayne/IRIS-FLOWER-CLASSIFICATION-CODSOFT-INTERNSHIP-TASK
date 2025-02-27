import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
def load_data():
    df = pd.read_csv(r"C:\Users\siddh\OneDrive\Desktop\codsoft intern\IRIS\IRIS.csv")
    return df

# Preprocess the data
def preprocess_data(df):
    X = df.drop('species', axis=1)
    y = df['species']
    return X, y

# Train the model
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

# Function to get user input
def get_user_input():
    print("\nEnter Iris flower measurements:")
    sepal_length = float(input("Sepal Length (cm): "))
    sepal_width = float(input("Sepal Width (cm): "))
    petal_length = float(input("Petal Length (cm): "))
    petal_width = float(input("Petal Width (cm): "))
    return np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Function to make prediction
def predict_species(model, scaler, user_input):
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)
    probability = model.predict_proba(user_input_scaled)
    return prediction[0], probability[0]

# Function to display visualizations
def display_visualizations(df, X, model):
    # Pairplot
    sns.pairplot(df, hue='species')
    plt.suptitle("Pairplot of Iris Features", y=1.02)
    plt.show()

    # Correlation Heatmap
    corr = df.drop('species', axis=1).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title("Feature Importance in Iris Classification")
    plt.show()

# Main function
def main():
    # Load and preprocess data
    df = load_data()
    X, y = preprocess_data(df)

    # Train model
    model, scaler, accuracy, report = train_model(X, y)

    print("Iris Flower Classification")
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(report)

    # Display visualizations
    display_visualizations(df, X, model)

    while True:
        user_input = get_user_input()
        prediction, probability = predict_species(model, scaler, user_input)

        print("\nPrediction Result:")
        print(f"The predicted Iris species is: {prediction}")
        print("\nPrediction Probabilities:")
        for species, prob in zip(model.classes_, probability):
            print(f"{species}: {prob:.4f}")

        again = input("\nWould you like to make another prediction? (yes/no): ")
        if again.lower() != 'yes':
            break

    print("Thank you for using the Iris Flower Classification model!")

if __name__ == "__main__":
    main()