from fastapi import FastAPI
from pydantic import BaseModel

class Data(BaseModel):
    x: float
    y: float
    
app = FastAPI()

@app.get("/")
async def index():
    return {"message": "Hello, Deta!"}

@app.post("/")
async def calc(data: Data):
    return {"result": data.x * data.y}
