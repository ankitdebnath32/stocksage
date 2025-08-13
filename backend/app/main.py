from fastapi import FastAPI
from .routes import prediction

app = FastAPI(title="Stocksage API")
app.include_router(prediction.router, prefix="/api/prediction")