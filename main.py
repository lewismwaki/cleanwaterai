#!/usr/bin/env python3

import sys, subprocess, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # run notebooks sequentially (optional not needed for streamlit to run)
    for i in range(1, 11):
        matches = list(Path(".").glob(f"notebooks/{i:02d}_*.ipynb"))
        if matches:
            subprocess.run(["jupyter", "nbconvert", "--execute", "--to", "notebook", "--inplace", str(matches[0])], check=True)
    
    # run scripts (required for streamlit to run effectively)
    for script in ["extract_data.py", "merge_data.py", "prepare_data.py", "train_xgb.py", "train_nlp.py"]:
        if Path(f"scripts/{script}").exists():
            subprocess.run([sys.executable, f"scripts/{script}"], check=True)
    
    # run apps (triggering live ingestion, predictions, and alerts)
    for app in ["trigger_ingestion.py", "trigger_predictions.py", "trigger_alerts.py"]:
        if Path(f"app/{app}").exists():
            subprocess.run([sys.executable, f"app/{app}"], check=True)
    
    # Start dashboard
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port", "8505"])

if __name__ == "__main__":
    main()
