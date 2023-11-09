# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("smoking_clf_api")

# Create input/output pydantic models
input_model = create_model("smoking_clf_api_input", **{'age': 60.0, 'height(cm)': 150.0, 'weight(kg)': 50.0, 'waist(cm)': 71.0, 'eyesight(left)': 0.8999999761581421, 'eyesight(right)': 1.0, 'hearing(left)': 1.0, 'hearing(right)': 1.0, 'systolic': 130.0, 'relaxation': 80.0, 'fasting blood sugar': 85.0, 'Cholesterol': 155.0, 'triglyceride': 89.0, 'HDL': 57.0, 'LDL': 80.0, 'hemoglobin': 13.0, 'Urine protein': 1.0, 'serum creatinine': 0.5, 'AST': 21.0, 'ALT': 10.0, 'Gtp': 16.0, 'dental caries': 0.0, 'bmi': 22.22222137451172, 'liver_enz': 47.0, 'totalDL': 137.0})
output_model = create_model("smoking_clf_api_output", prediction=1)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
