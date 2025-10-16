import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# =====================
# 1️⃣ Example Training Data
data = {
    'feature1': [5.1, 4.9, 6.2, 5.9, 5.5, 6.7, 5.8],
    'feature2': [3.5, 3.0, 3.4, 3.0, 3.6, 3.1, 2.7],
    'feature3': [1.4, 1.4, 5.4, 5.1, 1.4, 5.6, 5.1],
    'feature4': [0.2, 0.2, 2.3, 1.8, 0.3, 2.4, 1.9],
    'target': [0, 0, 1, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

X = df[['feature1','feature2','feature3','feature4']]
y = df['target']

# =====================
# 2️⃣ Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for future use
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# =====================
# 3️⃣ Train Model
model = RandomForestClassifier()
model.fit(X_scaled, y)

# Save model for future use
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# =====================
# 4️⃣ Prediction Function
def make_prediction(input_list):
    input_df = pd.DataFrame(input_list, columns=['feature1','feature2','feature3','feature4'])
    scaled_input = scaler.transform(input_df)
    return model.predict(scaled_input)

# =====================
# 5️⃣ Test Predictions
if __name__ == '__main__':
    test_input1 = [[5.1, 3.5, 1.4, 0.2]]
    test_input2 = [[6.2, 3.4, 5.4, 2.3]]
    print('Prediction 1:', make_prediction(test_input1))
    print('Prediction 2:', make_prediction(test_input2))
