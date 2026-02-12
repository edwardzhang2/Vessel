"""
Concurrent multi-GPU extractor: schedule per-question tasks dynamically across GPUs.

- One worker process per GPU.
- Each worker loads model once (initializer) and answers tasks.
- Parent process creates tasks for (pdf, page, question) and uses dynamic scheduling:
  submit slower tasks first, then keep GPUs fed until done.
- Parent assembles results into unified results.csv:
  FormType, File, then MD537_* and MD538_* columns.
"""

import os
import sys
import csv
import re
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------- logging -----------------
LOG_LEVEL = os.environ.get("LLAMA_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("llama_parallel")

# ----------------- configuration -----------------
os.environ.setdefault("HF_HOME", "/cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/cache")
LLAMA_MODEL_DIR = os.environ.get("LLAMA_MODEL_DIR", "/models/Llama-3.2-11B-Vision-Instruct")

MAX_SIZE_PAGE = int(os.environ.get("LLAMA_MAX_SIZE", "700"))
MAX_NEW_TOKENS = int(os.environ.get("LLAMA_MAX_NEW_TOKENS", "60"))

# Heuristic: expected relative cost per question index (1-based).
# This is to prevent the pipeline from getting stuck on slow tasks.
DEFAULT_COSTS: Dict[str, List[int]] = {
    "MD537_P1": [1,1,1,3,3,3,3,1,1,1,3],
    "MD537_P2": [10,20,20],
    "MD538_P1": [1,1,1,3,3,3,3,1,1,1,3],
    "MD538_P2": [10,20,15,10,2,2], # extracting long text takes significantly longer than short fields
    "MD538_P3": [2,2,2],
}

# ----------------- question sets -----------------
MD537_P1 = [
    "Based on the image above, output only the name of vessel and nothing else.",
    "Based on the image above, output only the call sign of vessel and nothing else.",
    "Based on the image above, output only the IMO number of vessel and nothing else.",
    "Based on the image above, output the handwritten answer to Question 4 which is 'Buoy and/or Anchorage(s) (by stating whether thes facilities 'will be used' / 'will not be used' during the stay in port). The answer will either be 'will be used' or 'will not be used'. Output nothing else.",
    "Based on the image above, output the availability of international ship security certificate (ISSC) or Interm ISSC. The answer is either 'Yes' or 'No'. Output nothing else.",
    "Based on the image above, output the expiry date of ISSC or Interim ISSC (in the form of 'YYYY/MM/DD'). Output nothing else.",
    "Based on the image above, output the issuing authority of ISSC or interim ISSC (by stating the name of the issuing authority). Output nothing else.",
    "Based on the image above, outpu the security level the ship is currently operating at (by stating 'Level 1', 'Level 2' or 'Level 3', with reason(s) if known). Output nothing else.",
    "Based on the image above, output the last port of call (by stating the name of the port) and nothing else.",
    "Based on the image above, output the name of the last port facility (by stating the name of the last port facility that the ship had interfaced with before departure) and nothing else.",
    "Based on the image above, output whether the last port facility is in compliance with ISPS (by stating 'Yes'/'No'). Output nothing else.",
]
MD537_P2 = [
    "Based on the image above, output the security level of last port facility (by stating 'Level 1', 'Level 2' or 'Level 3'. Output nothing else.",
    "Based on the image above, output whether for the last 10 calls at port facilities since 1st July 2004, has the ship interface with a port facility that was non-ISPS compliant (by stating 'Yes'/'No'). Output nothing else.",
    "Based on the image above, output whether within the period of the last 10 calls at the port facilities, has teh ship engaged in ship-to-ship activities with a non-compliant ship to which the ISPS code applies (by stating 'Yes'/'No'). Output nothing else.",
]
MD538_P1 = [
    "Based on the image above, output only the name of vessel and nothing else.",
    "Based on the image above, output only the call sign of vessel and nothing else.",
    "Based on the image above, output only the national colors of vessel and nothing else.",
    "Based on the image above, output only the length overall of vessel in meters and nothing else.",
    "Based on the image above, output only the maximum draft of vessel and nothing else.",
    "Based on the image above, output only the deadweight tonnage under present condtiion and nothing else.",
    "Based on the image above, output only the date keel laid and nothing else.",
    "Based on the image above, output only the cargo type and amount of cargo in tonnes (whether for loading, discharge, transshipment or transit) and nothing else.",
    "Based on the image above, output only the estimated time of arrival at intended pilot boarding station, specific anchorage or berth in the waters of Hong Kong (expressed as 'YY/MM/DD/hh/mm') and nothing else.",
    "Based on the image above, output only the estimated time of departure from intended berth in the waters of Hong Kong (expressed as 'YY/MM/DD/hh/mm') and nothing else. Make sure the answer is not the same as the one in item 9",
    "Based on the image above, output only the intended berth and nothing else.",
]
MD538_P2 = [
    "From the scanned document image above, determine whether there is any handwritten text inside the boxed area for item 12 and nothing else.",
    "From the scanned document image above, determine whether there is any handwritten text inside the boxed area for item 13 and nothing else.",
    "From the scanned document image above, determine whether there is any handwritten text inside the boxed area for item 14 and nothing else.",
    "From the scanned document image above, extract ONLY the handwritten text inside the boxed area for item 15 and nothing else.",
    "From the scanned document image above, extract ONLY the handwritten text inside the boxed area for item 16 and nothing else.",
    "From the scanned document image above, extract ONLY the handwritten text inside the boxed area for item 17 and nothing else.",
]
MD538_P3 = [
    "From the scanned document image above, extract ONLY the handwritten tick marks inside the boxed area for item 18(a) and nothing else.",
    "From the scanned document image above, extract ONLY the handwritten date inside the boxed area for item 18(b) and nothing else.",
    "From the scanned document image above, extract ONLY the handwritten tick mark inside the boxed area for item 18(c) and nothing else.",
]
# ----------------- unified CSV headers -----------------
def headers_unified() -> List[str]:
    md537_p1 = [f"MD537_P1_Q{i}" for i in range(1, 12)]
    md537_p2 = [f"MD537_P2_Q{i}" for i in range(1, 4)]
    md538_p1 = [f"MD538_P1_Q{i}" for i in range(1, 12)]
    md538_p2 = [f"MD538_P2_Q{i}" for i in range(1, 7)]
    md538_p3 = [f"MD538_P3_Q{i}" for i in range(1, 4)]
    return ["FormType", "File"] + md537_p1 + md537_p2 + md538_p1 + md538_p2 + md538_p3

HEADER = headers_unified()
# index of the headers in the csv file, O(1) lookup for faster access
HEADER_INDEX = {k: i for i, k in enumerate(HEADER)}

# finding all the PDFs in the folder
def get_pdfs(folder_path: str) -> List[str]:
    pdfs = []
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, fname))
    return sorted(pdfs)

# ----------------- worker globals -----------------
_WORKER_MODEL = None
_WORKER_PROCESSOR = None
_WORKER_DEVICE = None

def _init_worker(model_path: str, gpu_id: int):
    """
    Runs once per worker process. Pins the process to one GPU (cuda:<gpu_id>),
    loads model+processor.
    """
    global _WORKER_MODEL, _WORKER_PROCESSOR, _WORKER_DEVICE

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    os.environ.setdefault("HF_HOME", "/cache")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/cache")

    if torch.cuda.is_available():
        _WORKER_DEVICE = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(_WORKER_DEVICE)
    else:
        _WORKER_DEVICE = torch.device("cpu")

    _WORKER_PROCESSOR = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    # Vision2Seq is the model type for the LLaMA model, kind of like an interface for the model
    _WORKER_MODEL = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if _WORKER_DEVICE.type == "cuda" else None,
        local_files_only=True,
    ).to(_WORKER_DEVICE)
    _WORKER_MODEL.eval()

def _pdf_page_to_image(pdf_path: str, page_number: int, max_size: int):
    import fitz
    from PIL import Image
    # some preprocessing is done to the PDF to make it more readable
    doc = fitz.open(pdf_path)
    try:
        if page_number >= len(doc):
            raise ValueError(f"PDF '{pdf_path}' only has {len(doc)} pages.")
        page = doc[page_number]
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        doc.close()

    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)))
    return img

def _crop_bottom_left(img, width_frac=0.5, height_frac=0.3):
    # crops the image to the bottom left corner to identify the form type
    w, h = img.size
    left = 0
    right = int(w * max(0.2, min(1.0, width_frac)))
    top = int(h * max(0.0, min(1.0, 1.0 - height_frac)))
    bottom = h
    return img.crop((left, top, right, bottom))

def _ask_on_image(image, prompt: str) -> str:
    """
    Inference call, uses global model/processor/device
    """
    global _WORKER_MODEL, _WORKER_PROCESSOR, _WORKER_DEVICE
    import torch

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    input_text = _WORKER_PROCESSOR.apply_chat_template(messages, add_generation_prompt=True)
    inputs = _WORKER_PROCESSOR(images=image, text=input_text, return_tensors="pt", padding=False)
    inputs = {k: v.to(_WORKER_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        out = _WORKER_MODEL.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, num_beams=1, do_sample=False)

    text = _WORKER_PROCESSOR.decode(out[0], skip_special_tokens=True)
    if text.startswith(input_text):
        text = text[len(input_text):].strip()

    pos = text.lower().find("assistant")
    if pos != -1:
        text = text[pos + len("assistant"):].strip()
    return text.strip()

# Task = (pdf, formtype, page_idx, prefix, q_idx_1based, prompt)
Task = Tuple[str, str, int, str, int, str]

def _detect_formtype_worker(pdf: str) -> str:
    """
    Detects the form type (md537 or md538) by rendering page 0, cropping, and asking a strict token.
    """
    img0 = _pdf_page_to_image(pdf, 0, max_size=900)
    cropped = _crop_bottom_left(img0, width_frac=0.45, height_frac=0.28)
    strict = "Look at this bottom-left corner of the form only. Answer exactly one token: MD537 or MD538."
    ans = _ask_on_image(cropped, strict).upper()
    if "537" in ans:
        return "MD537"
    if "538" in ans:
        return "MD538"
    # fallback: page count heuristic
    try:
        import fitz
        doc = fitz.open(pdf)
        n_pages = len(doc)
        doc.close()
        return "MD537" if n_pages <= 2 else "MD538"
    except Exception:
        return "MD538"

def _run_task(task: Task) -> Tuple[str, str, str, int, str]:
    """
    Executes one question inference.
    Returns (pdf, prefix, answer, q_idx_1based, formtype)
    """
    pdf, formtype, page_idx, prefix, q_idx, prompt = task
    img = _pdf_page_to_image(pdf, page_idx, max_size=MAX_SIZE_PAGE)
    ans = _ask_on_image(img, prompt)
    return (pdf, prefix, ans, q_idx, formtype)

# ----------------- parent: build tasks -----------------
def _tasks_for_pdf(pdf: str, formtype: str) -> List[Task]:
    tasks: List[Task] = []
    if formtype == "MD537":
        # page 0 => P1, page 1 => P2
        for i, q in enumerate(MD537_P1, start=1):
            tasks.append((pdf, formtype, 0, "MD537_P1", i, q))
        for i, q in enumerate(MD537_P2, start=1):
            tasks.append((pdf, formtype, 1, "MD537_P2", i, q))
    else:
        for i, q in enumerate(MD538_P1, start=1):
            tasks.append((pdf, formtype, 0, "MD538_P1", i, q))
        for i, q in enumerate(MD538_P2, start=1):
            tasks.append((pdf, formtype, 1, "MD538_P2", i, q))
        for i, q in enumerate(MD538_P3, start=1):
            tasks.append((pdf, formtype, 2, "MD538_P3", i, q))
    return tasks

def _expected_cost(prefix: str, q_idx_1based: int) -> int:
    costs = DEFAULT_COSTS.get(prefix)
    i = q_idx_1based - 1
    return costs[i]

def _place_answer(row: List[str], prefix: str, q_idx_1based: int, answer: str):
    col = f"{prefix}_Q{q_idx_1based}"
    if col not in HEADER_INDEX:
        return
    row[HEADER_INDEX[col]] = answer or ""

# ----------------- main -----------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python llama_11b_parallel.py <pdf_folder_path>")
        sys.exit(1)

    input_folder = Path(sys.argv[1]).resolve()
    if not input_folder.is_dir():
        raise ValueError(f"Input path is not a directory: {input_folder}")

    # stub mode
    if os.environ.get("SKIP_LLAMA", "0") == "1":
        out_path = Path.cwd() / "results.csv"
        pdfs = sorted(input_folder.rglob("*.pdf"))
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            for pdf in pdfs:
                w.writerow(["MD538", str(pdf)] + [""] * (len(HEADER) - 2))
        logger.warning(f"[STUB] wrote {len(pdfs)} row(s) to {out_path}")
        return

    model_path = Path(LLAMA_MODEL_DIR)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    pdf_files = get_pdfs(str(input_folder))
    logger.info(f"Found {len(pdf_files)} PDF(s).")
    out_path = Path.cwd() / "results.csv"

    if not pdf_files:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADER)
        logger.info("No PDFs; wrote header-only results.csv")
        return

    # detect GPUs
    try:
        import torch
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        n_gpus = 0

    gpu_ids = list(range(n_gpus)) if n_gpus > 0 else [0]
    logger.info(f"Using {len(gpu_ids)} worker(s). GPU IDs: {gpu_ids}")

    # ceate one executor per GPU (max_workers=1 each), so each worker binds to one GPU.
    # having more than 1 max_worker leads to OOM errors 
    executors: List[ProcessPoolExecutor] = []
    for gid in gpu_ids:
        ex = ProcessPoolExecutor(
            max_workers=1,
            initializer=_init_worker,
            initargs=(str(model_path), gid),
        )
        executors.append(ex)

    # 1) detect form type for each PDF (parallel)
    detect_futs = []
    for i, pdf in enumerate(pdf_files):
        ex = executors[i % len(executors)]
        detect_futs.append(ex.submit(_detect_formtype_worker, pdf))

    pdf_to_form: Dict[str, str] = {}
    for pdf, fut in zip(pdf_files, detect_futs):
        # in production, would need to handle exceptions here
        pdf_to_form[pdf] = fut.result()

    md537_n = sum(1 for v in pdf_to_form.values() if v == "MD537")
    md538_n = sum(1 for v in pdf_to_form.values() if v == "MD538")
    logger.info(f"Detected FormType: MD537={md537_n}, MD538={md538_n}")

    # prepare output rows
    # row is full HEADER length, but we'll write with csv.writer (list)
    pdf_to_row: Dict[str, List[str]] = {}
    for pdf in pdf_files:
        row = [""] * len(HEADER)
        row[HEADER_INDEX["FormType"]] = pdf_to_form[pdf]
        row[HEADER_INDEX["File"]] = pdf
        pdf_to_row[pdf] = row

    # 2) build all tasks
    all_tasks: List[Task] = []
    for pdf in pdf_files:
        all_tasks.extend(_tasks_for_pdf(pdf, pdf_to_form[pdf]))

    # 3) do the slower tasks first
    all_tasks.sort(key=lambda t: _expected_cost(t[3], t[4]), reverse=True)

    # 4) dynamic scheduling: keep at most one in-flight task per GPU (since each executor has 1 worker)
    in_flight = {}
    task_iter = iter(all_tasks)

    def _submit_next(executor_index: int) -> bool:
        try:
            task = next(task_iter)
        except StopIteration:
            return False
        fut = executors[executor_index].submit(_run_task, task)
        in_flight[fut] = executor_index
        return True

    # prime the pipeline: one task per GPU
    for idx in range(len(executors)):
        _submit_next(idx)

    completed = 0
    t0 = time.time()

    # inflight[future object] = executor_index
    # future object is a placeholder for the currently running task's result

    try:
        # keep running until all tasks are complete
        while in_flight:
            # as_completed is a generator that yields completed future objects in the order they complete
            # list(in_flight.keys()) is a list of all the future objects currently in flight
            # we use list() so that it takes a snapshot, otherwise we would get a RuntimeError: dictionary changed size during iteration
            # timeout=None means no timeout, so as_completed will block until a future completes

            for fut in as_completed(list(in_flight.keys()), timeout=None):
                # get the executor index from the future object
                ex_idx = in_flight.pop(fut)

                # in production, would need to handle exceptions here
                pdf, prefix, ans, q_idx, formtype = fut.result()

                # store answer
                row = pdf_to_row[pdf]
                _place_answer(row, prefix, q_idx, ans)

                completed += 1
                if completed % 25 == 0:
                    logger.info(f"Completed {completed}/{len(all_tasks)} tasks...")

                # submit another task to this same executor (GPU) since this one just finished
                _submit_next(ex_idx)

                # break to refresh as_completed list (keeps responsiveness)
                break
    finally:
        # shutdown the executors
        for ex in executors:
            ex.shutdown(wait=True, cancel_futures=False)

    dt = time.time() - t0
    logger.info(f"All tasks done: {completed} tasks in {dt:.1f}s")

    # 5) write results.csv
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for pdf in pdf_files:
            w.writerow(pdf_to_row[pdf])

    logger.info(f"Wrote results.csv to: {out_path}")

if __name__ == "__main__":
    main()
