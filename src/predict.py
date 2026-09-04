import pandas as pd

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/Sales_Prediction_Model@champion")

new_data = pd.DataFrame({
    'TV': [35],
    'radio': [50000],
    'newspaper': [0]
})

prediction = model.predict(new_data)

print("Prediction:", prediction[0])