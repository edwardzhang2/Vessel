# pipeline.py
import os
import sys
import subprocess
import argparse
import uuid
import zipfile
import shutil
import csv
from pathlib import Path
from datetime import datetime

APP_DIR = Path(__file__).resolve().parent

LLAMA_SCRIPT = APP_DIR / "llama_11b.py"
EXTRACT_SCRIPT = APP_DIR / "extract_data.py"
CLASSIFY_SCRIPT = APP_DIR / "classify.py"

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/model.joblib")
FEATURES_PATH = os.environ.get("FEATURES_PATH", "/models/feature_columns.joblib")

# Prefer zipping over loose CSVs (keeps server from picking CSV)
PREFER_ZIP = os.environ.get("PIPELINE_PREFER_ZIP", "1") not in ("0", "false", "False")
# Remove loose CSVs after zipping (but keep them *inside* the ZIP)
CLEAN_AFTER_ZIP = os.environ.get("PIPELINE_CLEAN_AFTER_ZIP", "1") not in ("0", "false", "False")

# ---------- logging ----------
def _log_line(log_fp: Path, text: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    with log_fp.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {text}\n")

def _log_block(log_fp: Path, title: str, content: str) -> None:
    border = "=" * 80
    _log_line(log_fp, border)
    _log_line(log_fp, title)
    _log_line(log_fp, border)
    for line in content.splitlines():
        with log_fp.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

def _run_and_capture(step_name: str, cmd: list[str], cwd: Path, pipeline_log: Path, env: dict | None = None) -> None:
    _log_block(pipeline_log, f"[{step_name}] START", "")
    _log_line(pipeline_log, f"[{step_name}] CWD: {cwd}")
    _log_line(pipeline_log, f"[{step_name}] CMD: {' '.join(cmd)}")
    for k in ("LLAMA_MODEL_DIR", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "SKIP_LLAMA"):
        if env and k in env:
            _log_line(pipeline_log, f"[{step_name}] ENV {k}={env[k]}")
        elif k in os.environ:
            _log_line(pipeline_log, f"[{step_name}] ENV {k}={os.environ[k]}")

    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), env=env if env is not None else None,
            capture_output=True, text=True, check=False,
        )
        _log_block(pipeline_log, f"[{step_name}] STDOUT", proc.stdout or "(no stdout)")
        _log_block(pipeline_log, f"[{step_name}] STDERR", proc.stderr or "(no stderr)")
        _log_line(pipeline_log, f"[{step_name}] RETURN CODE: {proc.returncode}")

        if proc.returncode != 0:
            _log_line(pipeline_log, f"[{step_name}] FAILED")
            raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
        _log_line(pipeline_log, f"[{step_name}] OK")
    except Exception as e:
        _log_line(pipeline_log, f"[{step_name}] EXCEPTION: {repr(e)}")
        raise

# ---------- results helpers ----------
def _results_form_counts(results_csv: Path) -> tuple[int, int]:
    if not results_csv.exists():
        return 0, 0
    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return 0, 0
    form_key = None
    for k in rows[0].keys():
        if k.lower() == "formtype":
            form_key = k
            break
    if form_key is None:
        return 0, len(rows)
    md537 = sum(1 for r in rows if str(r.get(form_key, "")).strip().upper().startswith("MD537"))
    md538 = sum(1 for r in rows if str(r.get(form_key, "")).strip().upper().startswith("MD538"))
    return md537, md538

# ---------- steps ----------
def run_llama(input_path: str, pipeline_log: Path):
    job_dir = Path(input_path).resolve().parent
    if not LLAMA_SCRIPT.exists():
        _log_line(pipeline_log, f"[llama] ERROR: script not found at {LLAMA_SCRIPT}")
        raise FileNotFoundError(f"LLaMA script not found at {LLAMA_SCRIPT}")

    env = dict(os.environ)
    env.setdefault("HF_HOME", "/cache")
    env.setdefault("HUGGINGFACE_HUB_CACHE", "/cache")

    cmd = [sys.executable, str(LLAMA_SCRIPT), input_path]
    _run_and_capture("llama_11b.py", cmd, cwd=job_dir, pipeline_log=pipeline_log, env=env)

def run_extract_data(job_dir: Path, pipeline_log: Path):
    if not EXTRACT_SCRIPT.exists():
        _log_line(pipeline_log, f"[extract] ERROR: script not found at {EXTRACT_SCRIPT}")
        raise FileNotFoundError(f"extract_data.py not found at {EXTRACT_SCRIPT}")
    cmd = [sys.executable, str(EXTRACT_SCRIPT)]
    _run_and_capture("extract_data.py", cmd, cwd=job_dir, pipeline_log=pipeline_log)

def run_classify(job_dir: Path, pipeline_log: Path) -> str:
    if not CLASSIFY_SCRIPT.exists():
        _log_line(pipeline_log, f"[classify] ERROR: script not found at {CLASSIFY_SCRIPT}")
        raise FileNotFoundError(f"classify.py not found at {CLASSIFY_SCRIPT}")
    input_csv = "input.csv"
    output_csv = f"output_{uuid.uuid4().hex[:8]}.csv"
    cmd = [
        sys.executable, str(CLASSIFY_SCRIPT),
        input_csv, output_csv,
        "--model_path", MODEL_PATH,
        "--features_path", FEATURES_PATH,
    ]
    _run_and_capture("classify.py", cmd, cwd=job_dir, pipeline_log=pipeline_log)
    return output_csv

# ---------- packaging ----------
def _copy_worker_logs_into_job(job_dir: Path, pipeline_log: Path) -> None:
    tmp_dir = Path("/tmp")
    try:
        for log_path in tmp_dir.glob("llama_11b_*.log"):
            dest = job_dir / log_path.name
            if dest.exists():
                dest.unlink(missing_ok=True)
            shutil.copy2(log_path, dest)
            _log_line(pipeline_log, f"[package] Copied worker log: {dest.name}")
    except Exception as e:
        _log_line(pipeline_log, f"[package] WARN: could not collect worker logs: {e}")

def _zip_outputs(job_dir: Path, pipeline_log: Path) -> Path:
    _copy_worker_logs_into_job(job_dir, pipeline_log)
    zip_name = f"job_artifacts_{uuid.uuid4().hex[:8]}.zip"
    zip_path = job_dir / zip_name
    patterns = ["results*.csv", "output_*.csv", "pipeline.log", "llama_11b_*.log"]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        added_any = False
        for pat in patterns:
            for p in job_dir.glob(pat):
                try:
                    zf.write(p, arcname=p.name)
                    _log_line(pipeline_log, f"[package] Added: {p.name}")
                    added_any = True
                except Exception as e:
                    _log_line(pipeline_log, f"[package] WARN: failed to add {p.name}: {e}")
        if not added_any:
            _log_line(pipeline_log, "[package] WARN: nothing matched to zip")

    _log_line(pipeline_log, f"[package] ZIP created: {zip_path}")

    # Remove loose CSVs so the server picks the ZIP
    if PREFER_ZIP and CLEAN_AFTER_ZIP:
        removed = 0
        for pat in ("results*.csv", "input.csv", "output_*.csv"):
            for p in job_dir.glob(pat):
                try:
                    p.unlink(missing_ok=True)
                    removed += 1
                except Exception as e:
                    _log_line(pipeline_log, f"[package] WARN: could not remove {p.name}: {e}")
        _log_line(pipeline_log, f"[package] Cleaned loose CSVs ({removed}) to prefer ZIP.")
    elif PREFER_ZIP:
        _log_line(pipeline_log, "[package] Prefer ZIP is on, but CLEAN_AFTER_ZIP is off (leaving CSVs).")

    return zip_path

# ---------- pipeline ----------
def pipeline(input_path: str) -> str:
    job_dir = Path(input_path).resolve().parent
    pipeline_log = job_dir / "pipeline.log"

    try:
        if pipeline_log.exists():
            pipeline_log.unlink()
    except Exception:
        pass

    _log_block(pipeline_log, "PIPELINE START", f"job_dir={job_dir}\ninput_path={input_path}")
    _log_line(pipeline_log, f"Python: {sys.executable}")
    _log_line(pipeline_log, f"Python version: {sys.version}")
    _log_line(pipeline_log, f"MODEL_PATH={MODEL_PATH}")
    _log_line(pipeline_log, f"FEATURES_PATH={FEATURES_PATH}")
    _log_line(pipeline_log, f"PIPELINE_PREFER_ZIP={int(PREFER_ZIP)}")
    _log_line(pipeline_log, f"PIPELINE_CLEAN_AFTER_ZIP={int(CLEAN_AFTER_ZIP)}")

    # 1) OCR
    run_llama(input_path, pipeline_log)

    # Decide whether to extract/classify (MD538 only)
    results_csv = job_dir / "results.csv"
    md537_count, md538_count = _results_form_counts(results_csv)
    _log_line(pipeline_log, f"[pipeline] Form counts => MD537={md537_count}, MD538={md538_count}")

    if md538_count == 0:
        _log_line(pipeline_log, "[pipeline] No MD538 rows. Skipping extract_data.py and classify.py.")
    else:
        run_extract_data(job_dir, pipeline_log)
        input_csv = job_dir / "input.csv"
        if input_csv.exists():
            try:
                with input_csv.open("r", encoding="utf-8") as f:
                    lines = [ln for ln in (l.strip() for l in f) if ln]
                if len(lines) > 1:
                    out_name = run_classify(job_dir, pipeline_log)
                    _log_line(pipeline_log, f"[pipeline] classify.py produced: {job_dir / out_name}")
                else:
                    _log_line(pipeline_log, "[pipeline] input.csv has header only. Skipping classify.py.")
            except Exception as e:
                _log_line(pipeline_log, f"[pipeline] WARN: Could not inspect input.csv: {e}")
        else:
            _log_line(pipeline_log, "[pipeline] input.csv not found; skipping classify.py.")

    _log_block(pipeline_log, "PIPELINE DONE", "Packaging artifacts next.")
    zip_path = _zip_outputs(job_dir, pipeline_log)
    return str(zip_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full berth classification pipeline on PDF(s)")
    parser.add_argument("input_path", help="Path to input PDF file or folder of PDFs")
    args = parser.parse_args()

    job_dir_cli = Path(args.input_path).resolve().parent
    pipeline_log_cli = job_dir_cli / "pipeline.log"

    try:
        final_zip = pipeline(args.input_path)
        print(f"\nPipeline completed successfully!")
        print(f"Download all artifacts here: {final_zip}")
        print(f"Pipeline log: {pipeline_log_cli}")
    except Exception as e:
        print(f"\nPIPELINE FAILED. See detailed log: {pipeline_log_cli}")
        print(f"Error: {repr(e)}")
        raise
