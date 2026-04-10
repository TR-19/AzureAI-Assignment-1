import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("student-mat.csv", sep=';')

print(df.head())  # for viva

# Create Pass/Fail column (G3 = final grade)
df["Pass"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

# Features and target
X = df[["studytime"]]   # you can also use more features later
y = df["Pass"]

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Prediction
pred = model.predict([[3]])
print("Prediction (0=Fail, 1=Pass):", pred[0])

# Accuracy
y_pred = model.predict(X)
print("Accuracy:", accuracy_score(y, y_pred))