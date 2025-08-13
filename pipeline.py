# pipeline.py (PATCH)
import os
import sys
import subprocess
import argparse
import uuid
import time

def run_llama(input_path):
    print(f"Running llama_11B.py on input: {input_path} ...")
    cmd = [sys.executable, "llama_11B.py", input_path]
    subprocess.run(cmd, check=True)
    print("llama_11B.py completed. 'results.csv' generated.")

def run_extract_data():
    print("Running extract_data.py ...")
    cmd = [sys.executable, "extract_data.py"]
    subprocess.run(cmd, check=True)
    print("extract_data.py completed. 'input.csv' generated.")

def run_classify():
    input_csv = "input.csv"
    output_csv = f"output_{uuid.uuid4().hex[:8]}.csv"

    # NEW: read from env, fall back to container defaults under /models
    model_path = os.getenv("MODEL_PATH", "/models/model.joblib")
    features_path = os.getenv("FEATURES_PATH", "/models/feature_columns.joblib")

    print(f"Running classify.py on {input_csv} ...")
    cmd = [
        sys.executable, "classify.py",
        input_csv, output_csv,
        "--model_path", model_path,
        "--features_path", features_path,
    ]
    subprocess.run(cmd, check=True)
    print(f"classify.py completed. Output saved to '{output_csv}'")
    return output_csv

def pipeline(input_path):
    run_llama(input_path)
    run_extract_data()
    output_csv = run_classify()
    return output_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full berth classification pipeline on PDF(s)")
    parser.add_argument("input_path", help="Path to input PDF file or folder of PDFs")
    args = parser.parse_args()

    start_time = time.time()
    final_csv = pipeline(args.input_path)
    elapsed = time.time() - start_time

    print(f"\nPipeline completed successfully! Final classified CSV: {final_csv}")
    print(f"Total elapsed time: {elapsed:.2f} seconds")
