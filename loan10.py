import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("loan.csv")

print(df.head())

# ==========================
# MANUAL ENCODING
# ==========================

df["Gender"] = df["Gender"].map({
    "Male": 0,
    "Female": 1
})

df["Married"] = df["Married"].map({
    "No": 0,
    "Yes": 1
})

df["Dependents"] = df["Dependents"].map({
    "0": 0,
    "1": 1,
    "2": 2,
    "3+": 3
})

df["Education"] = df["Education"].map({
    "Not Graduate": 0,
    "Graduate": 1
})

df["Self_Employed"] = df["Self_Employed"].map({
    "No": 0,
    "Yes": 1
})

df["Property_Area"] = df["Property_Area"].map({
    "Rural": 0,
    "Semiurban": 1,
    "Urban": 2
})

df["Loan_Status"] = df["Loan_Status"].map({
    "Rejected": 0,
    "Approved": 1
})

# ==========================
# FEATURES & TARGET
# ==========================

x = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# ==========================
# TRAIN TEST SPLIT
# ==========================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# MODEL
# ==========================

model = GaussianNB()

model.fit(x_train, y_train)

# ==========================
# PREDICTION
# ==========================

pred = model.predict(x_test)

# ==========================
# ACCURACY
# ==========================

acc = accuracy_score(y_test, pred)

print("\nAccuracy :", round(acc * 100, 2), "%")

# ==========================
# CLASSIFICATION REPORT
# ==========================

print("\nClassification Report\n")
print(classification_report(y_test, pred))

# ==========================
# CONFUSION MATRIX
# ==========================

cm = confusion_matrix(y_test, pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.show()

# ==========================
# USER INPUT PREDICTION
# ==========================

print("\nEnter Applicant Details\n")

gender = int(input("Gender (Male=0 Female=1): "))
married = int(input("Married (No=0 Yes=1): "))
dependents = int(input("Dependents (0/1/2/3): "))
education = int(input("Education (Not Graduate=0 Graduate=1): "))
self_emp = int(input("Self Employed (No=0 Yes=1): "))

app_income = float(input("Applicant Income: "))
coapp_income = float(input("Coapplicant Income: "))

loan_amount = float(input("Loan Amount: "))
loan_term = float(input("Loan Term: "))

credit_history = int(input("Credit History (0/1): "))

property_area = int(
    input("Property Area (Rural=0 Semiurban=1 Urban=2): ")
)

person = [[
    gender,
    married,
    dependents,
    education,
    self_emp,
    app_income,
    coapp_income,
    loan_amount,
    loan_term,
    credit_history,
    property_area
]]

result = model.predict(person)

if result[0] == 1:
    print("\nLoan Approved ✅")
else:
    print("\nLoan Rejected ❌")
