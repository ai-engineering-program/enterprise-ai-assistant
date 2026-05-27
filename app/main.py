from fastapi import FastAPI

app = FastAPI(title="Enterprise AI Assistant", version="0.1.0")


@app.get("/health")
async def health():
    """Health check endpoint — always returns 200 OK."""
    return {"status": "ok", "version": "0.1.0"}
