from fastapi import FastAPI

from app.api.routes import router as api_router

app = FastAPI(title="Enterprise AI Assistant", version="0.1.0")
app.include_router(api_router)


@app.get("/health")
async def health():
    """Health check endpoint — always returns 200 OK."""
    return {"status": "ok", "version": "0.1.0"}
