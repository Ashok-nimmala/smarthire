
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split


data = data = pd.read_csv("crop_data.csv")

data.head()

data.shape

data.isnull().sum()


from sklearn.tree import DecisionTreeClassifier


# Load dataset
data = data = pd.read_csv("crop_data.csv")   # change to your dataset file

# Features and target
X = data.drop(["label", "Unnamed: 8", "Unnamed: 9"], axis=1)     # input features, drop 'Unnamed' columns
y = data["label"]                  # output label

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Decision Tree model
model = DecisionTreeClassifier()

# Train model
model.fit(X_train, y_train)

pickle.dump(model,open("model.pkl",'wb'))
print("Model saved successfully")