# =====================================================================
#  FMR LIVE BACKEND — paste this whole cell into Google Colab
#  Runtime -> Change runtime type -> GPU (T4)  BEFORE running.
#  It installs deps, opens an ngrok tunnel, and serves api.py (MedVLM-R1).
#  The public https URL it prints is what you paste into the dashboard.
# =====================================================================

# --- 1. Dependencies (torch/PIL already ship on Colab GPU runtimes) ---
!pip install -q "fastapi" "uvicorn[standard]" pyngrok python-multipart nest_asyncio
!pip install -q "transformers>=4.49" accelerate qwen-vl-utils "huggingface_hub>=0.34"

# --- 2. Get the thesis repo (skip the clone if you've mounted Drive) ---
import os
REPO_URL = "https://github.com/<you>/<your-fmr-repo>.git"   # <-- EDIT ME
REPO_DIR = "/content/btppp"                                  # clone target
FMR_DIR  = f"{REPO_DIR}/fmr"                                 # holds api.py + src/

if not os.path.isdir(FMR_DIR):
    !git clone {REPO_URL} {REPO_DIR}
assert os.path.isdir(FMR_DIR), f"{FMR_DIR} not found — fix REPO_URL / REPO_DIR."

# --- 3. ngrok auth token (grab it from https://dashboard.ngrok.com) ---
from pyngrok import ngrok, conf
NGROK_AUTH_TOKEN = "PASTE_YOUR_NGROK_AUTHTOKEN_HERE"         # <-- EDIT ME
conf.get_default().auth_token = NGROK_AUTH_TOKEN

# --- 4. Optional overrides (defaults come from api.py / experiment.yaml) ---
os.environ["FMR_MODEL_ID"]     = "JZPeterPan/MedVLM-R1"
os.environ["FMR_N_CONSISTENCY"] = "5"      # the 5 self-consistency passes
os.environ["FMR_ALPHA"]        = "0.15"    # conformal retained-error target

# --- 5. Import the FastAPI app defined in fmr/api.py ------------------
import sys
sys.path.insert(0, FMR_DIR)                 # so `import api` finds api.py
os.chdir(FMR_DIR)                           # so its relative paths resolve
from api import app                         # noqa: E402

# --- 6. Open the tunnel, print the URL, then serve -------------------
PORT = 8000
public_url = ngrok.connect(PORT, "http").public_url
print("\n" + "=" * 64)
print(f"  ✅  PUBLIC API URL:  {public_url}")
print(f"      Paste this into the dashboard's 'Live Demo' tab.")
print(f"      Quick test:      {public_url}/health")
print("=" * 64 + "\n")

import nest_asyncio, uvicorn        # nest_asyncio lets uvicorn run inside Colab's loop
nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=PORT)   # blocks this cell — keep it running
