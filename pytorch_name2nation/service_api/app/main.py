import sys
from fastapi import FastAPI
from name2lang import NameToNationClassify

version = f"{sys.version_info.major}.{sys.version_info.minor}"
app = FastAPI()

@app.get("/")
async def read_root():
   message = f"Hello World!"
   return {
      "message": message
   }

@app.get("/api/name2nation/{name}")
def read_item(name: str):
   name = name.strip()
   model = NameToNationClassify("../models")
   model.load_state()
   rc = model.predict(name)
   return {
      "langs": rc,
   }


   