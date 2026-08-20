import pickle
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 1. AI Dimaag ko load karna
model = pickle.load(open('loan_model.pkl', 'rb'))


@app.route('/')
def home():
  return 'Bhai, Backend Server ekdum mast chal raha hai!'


# 2. Prediction route
@app.route('/predict', methods=['POST'])
def predict():
  try:
    data = request.get_json()

    features = [
        float(data['annual_income']),
        float(data['debt_to_income_ratio']),
        float(data['credit_score']),
        float(data['loan_amount']),
        float(data['interest_rate']),
        int(data['gender']),
        int(data['marital_status']),
        int(data['education_level']),
        int(data['employment_status']),
        int(data['loan_purpose']),
        int(data['grade_subgrade']),
    ]

    prediction = model.predict([features])
    result = int(prediction[0])

    if result == 1:
      status = 'Congratulations! Your Loan is Approved. 🎉'
    else:
      status = 'Sorry, your Loan Application is Rejected (High Risk). ❌'

    return jsonify({'prediction_text': status})

  except Exception as e:
    return jsonify({'error': str(e)})


if __name__ == '__main__':
  app.run(debug=True)