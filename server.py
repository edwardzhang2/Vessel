from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
import tempfile, shutil, os, zipfile, subprocess, glob, uuid, textwrap

APP_TITLE = "PDF Processing API (Upload ZIP → CSV)"
PIPELINE_SCRIPT = "pipeline.py"   # runs llama_11b.py -> extract_data.py -> classify.py

app = FastAPI(title=APP_TITLE)


@app.get("/", response_class=HTMLResponse)
def home():
    # simple HTML form so users can upload a ZIP from a browser
    return textwrap.dedent("""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width,initial-scale=1"/>
      <title>Upload PDFs (ZIP) → CSV</title>
      <style>
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; }
        .card { max-width: 720px; padding: 1.25rem 1.5rem; border: 1px solid #e5e7eb; border-radius: 12px; }
        h1 { margin-top: 0; font-size: 1.25rem; }
        input[type=file] { margin: 1rem 0; }
        .hint { color: #6b7280; font-size: 0.95rem; }
        button { background: black; color: white; border: 0; padding: 0.6rem 1rem; border-radius: 8px; cursor: pointer; }
        button:hover { opacity: 0.9; }
        .footer { margin-top: 2rem; color: #6b7280; font-size: 0.85rem; }
        code { background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }
      </style>
    </head>
    <body>
      <div class="card">
        <h1>Upload a <code>.zip</code> of PDFs → get a CSV</h1>
        <p class="hint">This will run the full pipeline (LLM OCR → data extraction → RF classification) and return a CSV file.</p>
        <form action="/upload/" method="post" enctype="multipart/form-data">
          <input type="file" name="file" accept=".zip" required />
          <div>
            <button type="submit">Process</button>
          </div>
        </form>
        <p class="hint">
          API doc: <a href="/docs">/docs</a> &nbsp;|&nbsp; Health: <a href="/healthz">/healthz</a>
        </p>
      </div>
      <div class="footer">
        <p>Tip: set <code>MODEL_PATH</code>, <code>FEATURES_PATH</code>, and (optionally) <code>LLAMA_MODEL_DIR</code> via environment variables.</p>
      </div>
    </body>
    </html>
    """)


@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"


@app.post("/upload/")
async def upload_pdf_zip(file: UploadFile = File(...)):
    # hard-reject non-zip uploads
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are accepted")

    # isolate each request in a temp workdir
    with tempfile.TemporaryDirectory(prefix="job_") as tmpdir:
        zip_path = os.path.join(tmpdir, file.filename)
        # save upload
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # extract
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(tmpdir)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid ZIP file")

        # run the pipeline *inside* tmpdir so outputs land there
        cmd = ["python", PIPELINE_SCRIPT, tmpdir]
        try:
            proc = subprocess.run(
                cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=3600  # 60 minutes
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Pipeline timed out.")

        # persist logs in tmpdir for debug (useful when viewing container logs)
        (open(os.path.join(tmpdir, "stdout.log"), "w")).write(proc.stdout or "")
        (open(os.path.join(tmpdir, "stderr.log"), "w")).write(proc.stderr or "")

        if proc.returncode != 0:
            # bubble up stderr for visibility
            raise HTTPException(status_code=500, detail=f"Pipeline failed:\n{proc.stderr}")

        # prefer final classification CSV (output_*.csv)
        outputs = sorted(glob.glob(os.path.join(tmpdir, "output_*.csv")))
        if outputs:
            path = outputs[-1]
            # give a deterministic download name for this request
            return FileResponse(path, filename=f"output_{uuid.uuid4().hex[:8]}.csv", media_type="text/csv")

        # else fall back to raw extraction results if classify was skipped
        results_path = os.path.join(tmpdir, "results.csv")
        if os.path.exists(results_path):
            return FileResponse(results_path, filename="results.csv", media_type="text/csv")

        raise HTTPException(status_code=500, detail="No CSV produced by the pipeline.")
