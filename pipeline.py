# pipeline.py
import os
import sys
import subprocess
import argparse
import uuid
import shutil
from datetime import datetime
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

# LLaMA log dir (should match llama_11b.py default)
LLAMA_LOG_DIR = Path(os.environ.get("LLAMA_LOG_DIR", "/app/logs"))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _append_block(log_file: Path, title: str, content: str) -> None:
    """Append a neat titled block into pipeline.log."""
    sep = "=" * 80
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"\n{sep}\n[{timestamp}] {title}\n{sep}\n")
        if content:
            f.write(content)
            if not content.endswith("\n"):
                f.write("\n")


def _run_and_capture(name: str, cmd: list[str], cwd: Path, pipeline_log: Path, env: dict | None = None) -> None:
    """Run a subprocess, capture stdout/stderr, and append to pipeline.log. Raises on non-zero exit."""
    _append_block(pipeline_log, f"RUN {name}", f"CWD: {cwd}\nCMD: {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=True,
            text=True,
            capture_output=True,
            env=env
        )
        if proc.stdout:
            _append_block(pipeline_log, f"{name} STDOUT", proc.stdout)
        if proc.stderr:
            _append_block(pipeline_log, f"{name} STDERR", proc.stderr)
    except subprocess.CalledProcessError as e:
        # Attach outputs to the log before re-raising
        if e.stdout:
            _append_block(pipeline_log, f"{name} STDOUT (on error)", e.stdout)
        if e.stderr:
            _append_block(pipeline_log, f"{name} STDERR (on error)", e.stderr)
        raise


def _snapshot_llama_logs() -> set[str]:
    """Snapshot current llama log filenames so we can detect new ones created by the run."""
    if not LLAMA_LOG_DIR.exists():
        return set()
    return {p.name for p in LLAMA_LOG_DIR.glob("llama_11b_*.log")}


def _collect_llama_logs(job_dir: Path, before: set[str], pipeline_log: Path) -> None:
    """Copy any *new* llama logs (created during this run) into <job_dir>/logs/ and note in pipeline.log."""
    if not LLAMA_LOG_DIR.exists():
        _append_block(pipeline_log, "LLAMA LOGS", f"No LLAMA_LOG_DIR found at {LLAMA_LOG_DIR}")
        return

    after = {p.name for p in LLAMA_LOG_DIR.glob("llama_11b_*.log")}
    new_names = sorted(list(after - before))
    dst_dir = job_dir / "logs"
    _ensure_dir(dst_dir)

    if not new_names:
        _append_block(pipeline_log, "LLAMA LOGS", "No new llama logs detected.")
        return

    copied = []
    for name in new_names:
        src = LLAMA_LOG_DIR / name
        if src.exists():
            dst = dst_dir / name
            try:
                shutil.copy2(src, dst)
                copied.append(str(dst))
            except Exception as ex:
                _append_block(pipeline_log, "LLAMA LOGS COPY ERROR", f"Failed to copy {src} -> {dst}: {ex}")

    if copied:
        _append_block(pipeline_log, "LLAMA LOGS COPIED", "\n".join(copied))
    else:
        _append_block(pipeline_log, "LLAMA LOGS", "Detected new log names but none copied (race or permissions).")


def run_llama(input_path: str, pipeline_log: Path):
    """
    Run llama_11b.py against the input folder of PDFs.
    We run with cwd=<job_dir> so results.csv is written into that job folder.
    Also capture any llama logs produced during the run and copy them into job_dir/logs.
    """
    job_dir = Path(input_path).resolve().parent
    if not LLAMA_SCRIPT.exists():
        raise FileNotFoundError(f"LLaMA script not found at {LLAMA_SCRIPT}")

    # Snapshot logs before
    before_logs = _snapshot_llama_logs()

    # Propagate current env; ensure LLAMA_LOG_DIR is set (so llama writes where we expect)
    env = os.environ.copy()
    env.setdefault("LLAMA_LOG_DIR", str(LLAMA_LOG_DIR))

    cmd = [sys.executable, str(LLAMA_SCRIPT), input_path]
    _run_and_capture("llama_11b.py", cmd, cwd=job_dir, pipeline_log=pipeline_log, env=env)

    # Collect new llama logs after run
    _collect_llama_logs(job_dir, before_logs, pipeline_log)


def run_extract_data(job_dir: Path, pipeline_log: Path):
    """
    Run extract_data.py in job_dir; it reads results.csv and writes input.csv there.
    """
    if not EXTRACT_SCRIPT.exists():
        raise FileNotFoundError(f"extract_data.py not found at {EXTRACT_SCRIPT}")
    cmd = [sys.executable, str(EXTRACT_SCRIPT)]
    _run_and_capture("extract_data.py", cmd, cwd=job_dir, pipeline_log=pipeline_log)


def run_classify(job_dir: Path, pipeline_log: Path) -> str:
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
    _run_and_capture("classify.py", cmd, cwd=job_dir, pipeline_log=pipeline_log)
    return output_csv


def pipeline(input_path: str) -> str:
    """
    Full pipeline:
      1) LLaMA extraction -> results.csv (in job_dir)
      2) Data extraction -> input.csv (in job_dir)
      3) Classification -> output_<id>.csv (in job_dir)
    Also writes pipeline.log in the job dir and copies any LLaMA logs into job_dir/logs/.
    Returns absolute path to the final output CSV.
    """
    job_dir = Path(input_path).resolve().parent
    _ensure_dir(job_dir)

    pipeline_log = job_dir / "pipeline.log"
    _append_block(
        pipeline_log,
        "PIPELINE START",
        f"APP_DIR={APP_DIR}\nJOB_DIR={job_dir}\nMODEL_PATH={MODEL_PATH}\nFEATURES_PATH={FEATURES_PATH}\n"
        f"LLAMA_SCRIPT={LLAMA_SCRIPT}\nEXTRACT_SCRIPT={EXTRACT_SCRIPT}\nCLASSIFY_SCRIPT={CLASSIFY_SCRIPT}\n"
        f"LLAMA_LOG_DIR={LLAMA_LOG_DIR}"
    )

    # Step 1
    run_llama(input_path, pipeline_log)

    # Step 2
    run_extract_data(job_dir, pipeline_log)

    # Step 3
    output_csv_name = run_classify(job_dir, pipeline_log)

    _append_block(pipeline_log, "PIPELINE END", f"Final CSV: {output_csv_name}")
    return str(job_dir / output_csv_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full berth classification pipeline on PDF(s)")
    parser.add_argument("input_path", help="Path to input PDF file or folder of PDFs")
    args = parser.parse_args()

    final_csv = pipeline(args.input_path)
    print(f"\nPipeline completed successfully! Final classified CSV: {final_csv}")
