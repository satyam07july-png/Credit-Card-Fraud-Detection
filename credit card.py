import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

# ================= FAST MODE =================
FAST_MODE = True

# ================= LOAD DATA =================
df = pd.read_csv("credit card data/creditcard.csv")

if FAST_MODE:
    df = df.sample(30000, random_state=42)

# ================= BASIC INFO =================
print(df.head())
print(df['Class'].value_counts())

# ================= VISUALIZATION =================
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Normal Transactions")
plt.show()

# ================= SPLIT =================
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================= SCALING =================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)   # ✅ FIXED

# ================= SMOTE =================
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

print("Before SMOTE:\n", y_train.value_counts())
print("After SMOTE:\n", pd.Series(y_train_res).value_counts())

# ================= MODELS =================

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_res, y_train_res)
y_pred_lr = lr.predict(X_test_scaled)

# Random Forest (Optimized)
rf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
rf.fit(X_train_res, y_train_res)
y_pred_rf = rf.predict(X_test_scaled)

# ================= EVALUATION =================
print("\nLogistic Regression")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

print("\nRandom Forest")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ================= MODEL COMPARISON =================
models = ['Logistic', 'Random Forest']

precision = [
    precision_score(y_test, y_pred_lr),
    precision_score(y_test, y_pred_rf)
]

recall = [
    recall_score(y_test, y_pred_lr),
    recall_score(y_test, y_pred_rf)
]

f1 = [
    f1_score(y_test, y_pred_lr),
    f1_score(y_test, y_pred_rf)
]

comparison = pd.DataFrame({
    'Model': models,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
})

print("\nModel Comparison:\n", comparison)

# ================= SAVE MODEL =================
joblib.dump(rf, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model & Scaler saved successfully!")