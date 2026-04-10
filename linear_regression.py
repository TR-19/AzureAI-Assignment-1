import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load datasetpip install pandas numpy scikit-learn matplotlib
df = pd.read_csv("Salary_Data.csv")

print(df.head())   # to show dataset (important for viva)

# Features and target
X = df[["YearsExperience"]]
y = df["Salary"]

# Model
model = LinearRegression()
model.fit(X, y)

# Prediction
pred = model.predict([[5]])
print("Predicted Salary for 5 years:", pred[0])

# Plot
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Linear Regression")
plt.show()