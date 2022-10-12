import sys
from fastapi import FastAPI

version = f"{sys.version_info.major}.{sys.version_info.minor}"
app = FastAPI()

@app.get("/")
async def read_root():
   message = f"Hello World!"
   return {
      "message": message
   }

@app.get("/items/{item_id}")   
def read_item(item_id: int, q: str=None):
   return {
      "item_id": item_id,
      "q": q
   }

   