from fastapi import FastAPI
from backend.api.routes import csp

app = FastAPI(title="Diagnosis CSP API")

app.include_router(csp.router)