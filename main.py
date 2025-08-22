from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="Orama X API")
origins = ["https://www.oramax.space", "https://oramax.space", "http://localhost:3000"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"],)
@app.get("/")
def root(): return {"status": "ok", "app": "Orama X", "message": "Hello from app.oramax.space!"}
