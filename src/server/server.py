import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from hydra import compose, initialize
from pathlib import Path

from src.esitmator.bayes_estimator import BayessianClassifier
from src.preprocessing.preprocessor import Preprocessor
from src.server.lifespan import lifespan
from src.server.routes import router

with initialize(config_path="../../conf", version_base=None):
    _cfg = compose(config_name="model")

app = FastAPI(lifespan=lifespan)
app.state.preprocessor = Preprocessor()
app.state.clf = BayessianClassifier(
    n_samples=_cfg.n_samples,
    num_warmup=_cfg.num_warmup,
    num_chains=_cfg.num_chains,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(router)

_frontend = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _frontend.exists():
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("src.server.server:app", host="0.0.0.0", port=8000, reload=False)
