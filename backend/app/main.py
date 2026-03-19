from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="A-Eye API")
app.include_router(router)