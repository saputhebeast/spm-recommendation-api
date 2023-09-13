from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd
from sklearn.neighbors import NearestNeighbors


app = FastAPI()

model_file = 'shoe_recommendation_model.joblib'
knn_model, encoder, data = load(model_file)


class Input(BaseModel):
    Brand: str
    Type: str
    Gender: str
    Color: str
    Material: str


@app.get("/recommend")
async def get_recommendations(inputs: Input):
    user_input = pd.DataFrame([inputs.dict()])
    user_encoded = pd.DataFrame(encoder.transform(user_input))
    distances, indices = knn_model.kneighbors(user_encoded, n_neighbors=5)
    recommended_shoes = data.iloc[indices[0]]
    recommended_shoes_list = recommended_shoes.to_dict(orient='records')

    return {"recommendations": recommended_shoes_list}
