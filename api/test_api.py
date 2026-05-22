
from fastapi import FastAPI

app = FastAPI(
        title="Test API",
    version="1.0"
)


@app.get("/")
def home():
    return {"message": "API fonctionne correctement "}

#tester api avec : uvicorn api.test_api:app --reload
