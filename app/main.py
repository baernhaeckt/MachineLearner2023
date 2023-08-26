import uvicorn

from app_factory import app_factory

app = app_factory()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5001, reload=True)
