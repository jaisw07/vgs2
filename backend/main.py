from fastapi import FastAPI

from backend.api.routes import (
    knowledge,
    inference,
    entropy,
    csp,
    nlp,
    logger,
    diagnosis
)

app = FastAPI(title="Full Diagnosis API")

# Attach all routers
app.include_router(knowledge.router)
app.include_router(inference.router)
app.include_router(entropy.router)
app.include_router(csp.router)
app.include_router(nlp.router)
app.include_router(logger.router)
app.include_router(diagnosis.router)