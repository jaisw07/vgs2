from fastapi import FastAPI
from backend.api.routes import (
    knowledge,
    inference,
    csp,
    entropy,
    nlp,
    logger,
    diagnosis,
    initializer
)

app = FastAPI(title="Diagnosis API")

app.include_router(knowledge.router)
app.include_router(inference.router)
app.include_router(csp.router)
app.include_router(entropy.router)
app.include_router(nlp.router)
app.include_router(logger.router)
app.include_router(diagnosis.router)
app.include_router(initializer.router)