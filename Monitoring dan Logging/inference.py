"""
inference.py
Script untuk melakukan inferensi menggunakan model MLflow yang sudah di-serve
"""

import requests
import json

def predict(data):
    url = "http://127.0.0.1:5001/invocations"
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        predictions = result.get('predictions', [])
        print(f"[RESULT] Prediksi: {predictions}")
        return predictions
    else:
        print(f"[ERROR] Status code: {response.status_code}")
        print(f"[ERROR] Response: {response.text}")
        return None

if __name__ == "__main__":
    sample_data = {
        "dataframe_records": [
            {
                "Pclass": -1.5658,
                "Sex": 0,
                "Age": -0.5924,
                "SibSp": 0,
                "Parch": 0,
                "Fare": -0.5024,
                "Embarked": 2,
                "FamilySize": -0.4743,
                "IsAlone": 1
            }
        ]
    }
    
    print("[INFO] Mengirim request ke model...")
    result = predict(sample_data)
    
    if result is not None:
        survived = "Survived" if result[0] >= 0.5 else "Not Survived"
        print(f"[INFO] Kesimpulan: {survived}")