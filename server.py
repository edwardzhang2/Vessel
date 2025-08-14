# server.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
import tempfile
import shutil
import os
import zipfile
import subprocess
import glob
from pathlib import Path
import sys

app = FastAPI(title="Vessel OCR & Classification API")

APP_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = APP_DIR / "pipeline.py"

HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Vessel OCR & Classification</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0f172a;--card:#111827;--text:#e5e7eb;--muted:#9ca3af;--accent:#22d3ee;}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(160deg,#0f172a,#0b1022);color:var(--text);font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell}
.container{max-width:900px;margin:6rem auto;padding:0 1rem}
.card{background:rgba(17,24,39,.7);backdrop-filter: blur(6px);border:1px solid rgba(255,255,255,.06);border-radius:18px;padding:2rem;box-shadow:0 10px 30px rgba(0,0,0,.35)}
h1{margin:0 0 .25rem;font-weight:800;letter-spacing:.3px}
.sub{color:var(--muted);margin:0 0 1.5rem}
label{display:block;margin:.5rem 0 .5rem;color:#cbd5e1}
input[type=file]{width:100%;padding:1rem;border:1px dashed rgba(255,255,255,.18);border-radius:12px;background:rgba(255,255,255,.03);color:var(--muted)}
button{margin-top:1rem;display:inline-flex;gap:.5rem;align-items:center;padding:.9rem 1.1rem;border-radius:12px;border:0;background:var(--accent);color:#001018;font-weight:700;cursor:pointer}
button:hover{filter:brightness(1.05)}
.help{margin-top:1rem;color:var(--muted);font-size:.95rem}
.footer{margin-top:2rem;color:#94a3b8;font-size:.85rem}
a{color:#7dd3fc;text-decoration:none}
a:hover{text-decoration:underline}
</style>
</head>
<body>
  <div class="container">
    <div class="card">
      <h1>Vessel OCR & Classification</h1>
      <p class="sub">Upload a <b>.zip</b> containing one or more PDFs. The pipeline will extract handwriting, normalize fields, and return a <b>CSV</b> with classifications.</p>
      <form action="/upload/" method="post" enctype="multipart/form-data">
        <label for="file">ZIP of PDFs</label>
        <input id="file" type="file" name="file" accept=".zip" required>
        <button type="submit">⏫ Upload &amp; Process</button>
      </form>
      <p class="help">Need API docs? See <a href="/docs" target="_blank">/docs</a>. Health check: <a href="/health" target="_blank">/health</a>.</p>
      <div class="footer">Runs on HPCaaS. Ensure model artifacts are mounted at <code>/models</code> for full classification.</div>
    </div>
  </div>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def root():
    return HTML_PAGE

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"

@app.post("/upload/")
async def upload_pdf_zip(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are accepted")

    if not PIPELINE_PATH.exists():
        raise HTTPException(status_code=500, detail=f"pipeline.py not found at {PIPELINE_PATH}")

    # Create an isolated job workspace under /tmp
    job_dir = Path(tempfile.mkdtemp(prefix="job_"))
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save the uploaded ZIP
        zip_path = job_dir / file.filename
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Extract into input_dir
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(input_dir)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid ZIP file")

        # Run pipeline.py with cwd=job_dir so outputs are written into job_dir
        cmd = [sys.executable, str(PIPELINE_PATH), str(input_dir)]
        proc = subprocess.run(
            cmd,
            cwd=str(job_dir),
            capture_output=True,
            text=True,
            timeout=3600
        )

        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Pipeline failed:\n{proc.stderr}")

        # Find the output CSV produced by pipeline (output_*.csv)
        outputs = sorted(job_dir.glob("output_*.csv"))
        if not outputs:
            # Fallback to results.csv if needed
            results = job_dir / "results.csv"
            if results.exists():
                outputs = [results]

        if not outputs:
            raise HTTPException(
                status_code=500,
                detail=f"No output CSV found in {job_dir}.\nStdout:\n{proc.stdout}\nStderr:\n{proc.stderr}"
            )

        out_csv = outputs[-1]
        return FileResponse(path=str(out_csv), filename=out_csv.name, media_type="text/csv")

    finally:
        # To keep artifacts for debugging, comment the next line out.
        # shutil.rmtree(job_dir, ignore_errors=True)
        pass
