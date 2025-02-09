import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the dataset (A-Z)
data_dict = pickle.load(open('./data_A_to_Z.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

# Initialize and train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# Predict and evaluate the model
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)

# Print evaluation metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_predict))

# Save the trained model
with open('model_A_to_Z.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved as 'model_A_to_Z.p'")
