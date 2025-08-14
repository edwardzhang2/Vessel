# pipeline.py
import os
import sys
import subprocess
import argparse
import uuid
from pathlib import Path

# Resolve the directory that contains this file (/app inside the image)
APP_DIR = Path(__file__).resolve().parent

# Script paths (match your actual filenames)
LLAMA_SCRIPT = APP_DIR / "llama_11b.py"        # NOTE: lowercase 'b' per your tree
EXTRACT_SCRIPT = APP_DIR / "extract_data.py"
CLASSIFY_SCRIPT = APP_DIR / "classify.py"

# Allow model paths to come from env (with sensible defaults)
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/model.joblib")
FEATURES_PATH = os.environ.get("FEATURES_PATH", "/models/feature_columns.joblib")


def run_llama(input_path: str):
    """
    Run llama_11b.py against the input folder of PDFs.
    We run with cwd=<job_dir> so results.csv is written into that job folder.
    """
    job_dir = Path(input_path).resolve().parent
    if not LLAMA_SCRIPT.exists():
        raise FileNotFoundError(f"LLaMA script not found at {LLAMA_SCRIPT}")
    cmd = [sys.executable, str(LLAMA_SCRIPT), input_path]
    subprocess.run(cmd, check=True, cwd=str(job_dir))


def run_extract_data(job_dir: Path):
    """
    Run extract_data.py in job_dir; it reads results.csv and writes input.csv there.
    """
    if not EXTRACT_SCRIPT.exists():
        raise FileNotFoundError(f"extract_data.py not found at {EXTRACT_SCRIPT}")
    cmd = [sys.executable, str(EXTRACT_SCRIPT)]
    subprocess.run(cmd, check=True, cwd=str(job_dir))


def run_classify(job_dir: Path) -> str:
    """
    Run classify.py on input.csv in job_dir; returns the output CSV filename.
    """
    if not CLASSIFY_SCRIPT.exists():
        raise FileNotFoundError(f"classify.py not found at {CLASSIFY_SCRIPT}")

    input_csv = "input.csv"
    output_csv = f"output_{uuid.uuid4().hex[:8]}.csv"

    cmd = [
        sys.executable,
        str(CLASSIFY_SCRIPT),
        input_csv,
        output_csv,
        "--model_path",
        MODEL_PATH,
        "--features_path",
        FEATURES_PATH,
    ]
    subprocess.run(cmd, check=True, cwd=str(job_dir))
    return output_csv


def pipeline(input_path: str) -> str:
    """
    Full pipeline:
      1) LLaMA extraction -> results.csv (in job_dir)
      2) Data extraction -> input.csv (in job_dir)
      3) Classification -> output_<id>.csv (in job_dir)
    Returns absolute path to the final output CSV.
    """
    job_dir = Path(input_path).resolve().parent

    # Step 1
    run_llama(input_path)

    # Step 2
    run_extract_data(job_dir)

    # Step 3
    output_csv_name = run_classify(job_dir)

    return str(job_dir / output_csv_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full berth classification pipeline on PDF(s)")
    parser.add_argument("input_path", help="Path to input PDF file or folder of PDFs")
    args = parser.parse_args()

    final_csv = pipeline(args.input_path)
    print(f"\nPipeline completed successfully! Final classified CSV: {final_csv}")
