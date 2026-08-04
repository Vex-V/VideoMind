"""Start the VideoMind server.

    python serve.py [--port 8077]

Sets the quieting env vars before anything imports transformers, then hands
off to uvicorn. `python -m uvicorn videomind.api.app:app` works too - app.py
sets the same vars - but this is the tidier entry point.
"""

import argparse
import os
import warnings

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    print(f"VideoMind -> http://{args.host}:{args.port}")
    uvicorn.run(
        "videomind.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
