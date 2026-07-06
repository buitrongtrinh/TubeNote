"""FastAPI entrypoint for TubeNote."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import dubbing, rag, video

app = FastAPI(title="TuBeNote API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mỗi feature 1 router — tách file ở backend/api/
app.include_router(video.router)
app.include_router(dubbing.router)
app.include_router(rag.router)


@app.get("/")
def root():
    return {"name": "TuBeNote API", "docs": "/docs"}
