import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from endpoint_funcs import load_model, predict, retrain
from typing import List
from datetime import datetime

# defining the main app
app = FastAPI(title="News article classifier", docs_url="/")

# calling the load_model during startup.
# this will train the model and keep it loaded for prediction.
app.add_event_handler("startup", load_model)

# class which is expected in the payload


class QueryIn(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# class which is returned in the response
class QueryOut(BaseModel):
    news_category: str
    timestamp: datetime

    # class which is expected in the payload while re-training


class FeedbackIn(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    flower_class: str

# Route definitions


@app.get("/ping")
# Healthcheck route to ensure that the API is up and running
def ping():
    return {"ping": "pong"}


@app.post("/predict_category", response_model=QueryOut, status_code=200)
# Route to do the prediction using the ML model defined.
# Payload: QueryIn containing the parameters
# Response: QueryOut containing the news_category predicted (200)
def predict_category(query_data: QueryIn):
    output = {"flower_class": predict(query_data), "timestamp": datetime.now()}
    return output


@app.post("/retrain_model", status_code=200)
# Route to retrain the model on live news data
# Response: Dict with detail confirming success (200)
def retrain_model():
    retrain()
    return {"detail": "Retraining successfull!"}


# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
