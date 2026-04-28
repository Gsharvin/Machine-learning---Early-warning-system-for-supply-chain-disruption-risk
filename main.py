import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.routes import router
from services import inference

app = FastAPI(title="Neptune API", description="Global Supply-Chain Risk Forecasting System")

# Allow Next.js frontend to communicate (still useful for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup: load ML models BEFORE any request is served
@app.on_event("startup")
async def startup_event():
    print("[Neptune] Loading ML models...")
    inference.load_models()
    print("[Neptune] Models loaded. Heatmap data ready.")

# Backend API routes
app.include_router(router, prefix="/api")

# Serve frontend static files
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "out")

if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    @app.get("/")
    def read_root():
        return {"status": "Neptune Core API is Online", "version": "1.0.0", "warning": "Frontend build not found"}

# SPA catch-all (optional for this app but robust)
@app.exception_handler(404)
async def custom_404_handler(request, __):
    if not request.url.path.startswith("/api"):
        index_file = os.path.join(frontend_path, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
    return {"detail": "Not Found"}
