import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

print("Data load aur process ho raha hai...")
# 1. Data load karna
df = pd.read_csv('train.csv')

# 2. Text ko Number mein badalna (Jo tumne abhi kiya)
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

# 3. Questions (X) aur Answer (y) ko alag karna
# 'loan_paid_back' humara answer hai, baaki sab questions hain. 'id' ka koi kaam nahi isliye use bhi hata rahe hain.
X = df.drop(columns=['loan_paid_back', 'id'], errors='ignore') 
y = df['loan_paid_back']

# 4. Model Training (Dimaag banana)
print("AI Model train ho raha hai... (Kripya pratiksha karein ⏳)")
model = RandomForestClassifier()
model.fit(X, y)

# 5. Model Save karna (Dimaag ko hamesha ke liye store karna)
with open('loan_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("--- Badhai Ho! Tumhara AI Dimaag ban gaya aur 'loan_model.pkl' naam se save ho gaya! 🎉 ---")