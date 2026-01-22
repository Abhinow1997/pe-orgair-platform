from fastapi import FastAPI
from datetime import datetime

app = FastAPI(title="Docker Lab API")

@app.get("/")
def root():
    return {"message": "Hello from Docker!", "status": "running"}

@app.get("/health")
def health():
    return {
        "healthy": True,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/greet/{name}")
def greet(name: str):
    return {"greeting": f"Hello, {name}! Welcome to Docker."}