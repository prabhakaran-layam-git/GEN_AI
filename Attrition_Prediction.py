import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.DataFrame({
    "Age":[29,35,40,28,45,25,50,30,37,26],
    "JobRole":["Sales Executive","Research Scientist","Laboratory Technician",
               "Sales Executive","Manager","Research Scientist","Manager",
               "Sales Executive","Laboratory Technician","Research Scientist"],
    "MonthlyIncome":[4800,6000,3400,4300,11000,3500,12000,5000,3100,4500],
    "JobSatisfaction":[3,4,2,3,4,1,4,2,2,3],
    "YearsAtCompany":[4,8,6,3,15,2,20,5,9,2],
    "Attrition":[1,0,0,1,0,1,0,0,0,1]
})

X = data.drop("Attrition", axis=1)
y = data["Attrition"]

# Preprocessing: encode categorical + scale numerical
numeric_features = ["Age","MonthlyIncome","JobSatisfaction","YearsAtCompany"]
categorical_features = ["JobRole"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(), categorical_features)
    ]
)

# Build pipeline
knn = KNeighborsClassifier(n_neighbors=3)
model = Pipeline(steps=[("preprocessor", preprocessor),
                       ("classifier", knn)])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Predict for user-entered data
allowed_roles = {
    "sales executive": "Sales Executive",
    "research scientist": "Research Scientist",
    "laboratory technician": "Laboratory Technician",
    "manager": "Manager"
}


def get_valid_role():
    while True:
        role = input("Job role (Sales Executive, Research Scientist, Laboratory Technician, Manager): ").strip()
        normalized_role = role.lower()
        if normalized_role in allowed_roles:
            return allowed_roles[normalized_role]
        print("Invalid job role. Please choose one of the listed options.")


print("\nEnter employee details to predict attrition:")
age = int(input("Age: "))
job_role = get_valid_role()
monthly_income = float(input("Monthly income: "))
job_satisfaction = int(input("Job satisfaction (1-4): "))
years_at_company = int(input("Years at company: "))

user_data = pd.DataFrame([{
    "Age": age,
    "JobRole": job_role,
    "MonthlyIncome": monthly_income,
    "JobSatisfaction": job_satisfaction,
    "YearsAtCompany": years_at_company
}])

prediction = model.predict(user_data)[0]
probability = model.predict_proba(user_data)[0]

if prediction == 1:
    result = "Likely to leave the company"
else:
    result = "Likely to stay with the company"

print(f"\nPrediction: {result}")
print(f"Attrition probability: {probability[1]:.2%}")
