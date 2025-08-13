from fastapi import FastAPI
from app.routes import prediction

app = FastAPI(title="StockSage API", version="0.1")
app.include_router(prediction.router, prefix="/api/prediction")