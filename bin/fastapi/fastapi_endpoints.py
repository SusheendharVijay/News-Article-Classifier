import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from ml_utils import train, predict, retrain
from typing import List
from datetime import datetime

# defining the main app
app = FastAPI(title="News article classifier", docs_url="/")

# calling the load_model during startup.
# this will train the model and keep it loaded for prediction.
app.add_event_handler("startup", train)

# class which is expected in the payload


class QueryIn(BaseModel):
    title: str
    description: str


# class which is returned in the response
class QueryOut(BaseModel):
    news_category: str
    timestamp: datetime

    # class which is expected in the payload while re-training


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
    output = {
        "news_category": predict(query_data),
        "timestamp": datetime.now()
    }
    print(output)
    return output


@app.get("/retrain_model", status_code=200)
# Route to retrain the model on live news data
# Response: Dict with detail confirming success (200)
def retrain_model():
    retrain()
    return {"detail": "Retraining successfull!"}


# Main function to start the app when fastapi_endpoints.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    uvicorn.run("fastapi_endpoints:app",
                host="0.0.0.0", port=8000, reload=True)
