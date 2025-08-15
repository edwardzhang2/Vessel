#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vessel OCR via Llama 3.2 11B Vision — parallel, per-GPU processing.

Features:
- Recursive PDF discovery (handles nested folders inside uploaded ZIPs)
- Detailed logging to stdout + per-process log file
- Env-driven model directory (LLAMA_MODEL_DIR), default: /models/Llama-3.2-11B-Vision-Instruct
- Local-only model & processor load (local_files_only=True). No online downloads.
- Optional SKIP_LLAMA=1 stub mode that writes a valid results.csv with 1 row per PDF (blank answers)
- Parallel per-GPU processing using ProcessPoolExecutor + per-page ThreadPoolExecutor
"""

import os
import sys
import csv
import time
import subprocess
import itertools
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# ---------------------------
# Logging setup (do this first so stub path also logs)
# ---------------------------
LOG_DIR = Path(os.environ.get("LLAMA_LOG_DIR", "/app/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"llama_11b_{os.getpid()}.log"
LOG_LEVEL = os.environ.get("LLAMA_LOG_LEVEL", "DEBUG").upper()

# Avoid duplicate handlers if module is re-imported by worker processes
root_logger = logging.getLogger()
if root_logger.handlers:
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.DEBUG),
    format="%(asctime)s [%(levelname)s] [pid=%(process)d] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),         # to container logs
        logging.FileHandler(LOG_FILE, mode="w"),   # per-process file log
    ],
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to {LOG_FILE}")

# ---------------------------
# Shared CSV header helper
# ---------------------------
def _headers():
    page1 = [f"Page1_Q{i}" for i in range(1, 12)]  # 11 items
    page2 = [f"Page2_Q{i}" for i in range(1, 7)]   # 6 items
    page3 = [f"Page3_Q{i}" for i in range(1, 4)]   # 3 items
    return ["File"] + page1 + page2 + page3

def _write_stub_results(input_folder: Path, out_path: Path):
    """Write a stub results.csv: 1 blank row per discovered PDF."""
    pdfs = sorted(input_folder.rglob("*.pdf"))
    hdrs = _headers()
    logger.warning(f"[STUB] Writing stub results for {len(pdfs)} PDF(s) to {out_path}")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hdrs)
        for pdf in pdfs:
            w.writerow([str(pdf)] + [""] * (len(hdrs) - 1))

# ---------------------------
# Quick STUB path (for smoke tests) — set SKIP_LLAMA=1
# ---------------------------
if os.environ.get("SKIP_LLAMA", "0") == "1":
    if len(sys.argv) < 2:
        logger.error("Usage: python llama_11b.py <pdf_folder_path>")
        sys.exit(0)
    in_dir = Path(sys.argv[1]).resolve()
    if not in_dir.is_dir():
        logger.error(f"Error: '{in_dir}' is not a directory.")
        sys.exit(0)
    out_file = Path.cwd() / "results.csv"
    logger.info(f"[STUB] Scanning PDFs in '{in_dir}'")
    _write_stub_results(in_dir, out_file)
    logger.info("[STUB] Done.")
    sys.exit(0)

# ---------------------------
# Heavy imports (only when not skipping)
# ---------------------------
import torch
from PIL import Image
import fitz  # PyMuPDF
from transformers import AutoModelForVision2Seq, AutoProcessor

# ---------------------------
# Config / defaults
# ---------------------------
os.environ.setdefault("HF_HOME", "/cache")              # preferred over TRANSFORMERS_CACHE
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/cache")

LLAMA_MODEL_DIR = os.environ.get("LLAMA_MODEL_DIR", "/models/Llama-3.2-11B-Vision-Instruct")

def print_gpu_usage():
    """Log GPU utilization and VRAM usage using nvidia-smi (best effort)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        logger.info("GPU Usage:")
        for line in result.stdout.strip().split("\n"):
            index, util, mem_used, mem_total = line.split(", ")
            logger.info(f"  GPU {index}: Utilization: {util}%  VRAM Used: {mem_used} MiB / {mem_total} MiB")
    except Exception as e:
        logger.warning(f"Could not get GPU usage info via nvidia-smi: {e}")

def ask_single_question_on_device(model, processor, image, question, device_id):
    """Ask one question about one image on a specific GPU device."""
    device = torch.device(f"cuda:{device_id}")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(images=image, text=input_text, return_tensors="pt", padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    torch.cuda.synchronize(device)
    start = time.time()
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=1,
            early_stopping=False,
            do_sample=False,
        )
        torch.cuda.synchronize(device)
        duration = time.time() - start

        output_text = processor.decode(outputs[0], skip_special_tokens=True)
        if output_text.startswith(input_text):
            output_text = output_text[len(input_text):].strip()
        pos = output_text.lower().find("assistant")
        if pos != -1:
            output_text = output_text[pos + len("assistant"):].strip()

        logger.debug(f"[GPU {device_id}] Q done in {duration:.2f}s; answer len={len(output_text)}")
        return question, output_text, duration

    except torch.cuda.OutOfMemoryError as oom:
        logger.error(f"[GPU {device_id}] CUDA OOM on question '{question}': {oom}")
        torch.cuda.empty_cache()
        return question, "Error: CUDA Out Of Memory", 0
    except Exception as e:
        logger.exception(f"[GPU {device_id}] Unexpected error on question '{question}': {e}")
        return question, f"Error: {e}", 0

def pdf_page_to_image(pdf_path, page_number, max_size=560):
    """Render a PDF page to a PIL.Image RGB with max edge <= max_size."""
    doc = fitz.open(pdf_path)
    if page_number >= len(doc):
        doc.close()
        raise ValueError(f"PDF '{pdf_path}' only has {len(doc)} pages.")
    page = doc[page_number]
    pix = page.get_pixmap(dpi=300)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size)
    return image

def write_results_to_csv(filepath, results, headers):
    with open(filepath, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in results:
            writer.writerow(row)
    logger.info(f"Wrote CSV: {filepath}")

def get_pdf_files_from_folder(folder_path):
    """Recursive scan for PDFs (handles nested folders inside a ZIP)."""
    pdfs = []
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, fname))
    logger.debug(f"Recursive scan complete under '{folder_path}': found {len(pdfs)} PDFs")
    for p in pdfs[:20]:
        logger.debug(f"  - {p}")
    return pdfs

def clean_llm_response(_question, raw_text):
    return raw_text.strip()

def process_single_pdf(pdf_file, model_path, max_size, all_questions_per_page, device_id):
    """
    Process one PDF file: load model on device, process each page and its questions in parallel threads,
    and return answers. This runs in a subprocess (spawned by ProcessPoolExecutor).
    """
    import torch as _torch
    import fitz as _fitz  # noqa: F401 (ensure availability in subprocess)
    from PIL import Image as _Image  # noqa: F401
    from transformers import AutoModelForVision2Seq as _AutoModel, AutoProcessor as _AutoProcessor

    pid = os.getpid()
    logger_local = logging.getLogger(f"worker-{pid}")
    logger_local.setLevel(logging.getLogger().level)
    logger_local.info(f"[{pdf_file}] Worker PID {pid} on GPU {device_id}")

    device = _torch.device(f"cuda:{device_id}")

    try:
        processor = _AutoProcessor.from_pretrained(model_path, local_files_only=True)
    except Exception as e:
        logger_local.exception(f"[{pdf_file}] Failed to load processor from '{model_path}': {e}")
        return pdf_file, []

    try:
        logger_local.info(f"[{pdf_file}] Loading model on GPU {device_id} ...")
        model = _AutoModel.from_pretrained(
            model_path,
            torch_dtype=_torch.bfloat16,
            device_map=None,         # place whole model on this device
            local_files_only=True,
        )
        model.to(device)
        model.eval()
        logger_local.info(f"[{pdf_file}] Model loaded on GPU {device_id}.")
    except Exception as e:
        logger_local.exception(f"[{pdf_file}] Failed to load model from '{model_path}': {e}")
        return pdf_file, []

    all_answers = []
    skip_pdf = False

    for page_num, questions in enumerate(all_questions_per_page):
        try:
            image = pdf_page_to_image(pdf_file, page_num, max_size)
        except Exception as e:
            logger_local.exception(f"[{pdf_file}] Failed to load page {page_num + 1}: {e}")
            skip_pdf = True
            break

        logger_local.debug(f"[{pdf_file}] Loaded page {page_num + 1}")

        page_answers = [None] * len(questions)
        with ThreadPoolExecutor(max_workers=len(questions)) as executor:
            futures = {
                executor.submit(
                    ask_single_question_on_device,
                    model, processor, image, question, device_id
                ): i for i, question in enumerate(questions)
            }

            for future in as_completed(futures):
                i = futures[future]
                try:
                    _q, answer, duration = future.result()
                except Exception as e:
                    logger_local.exception(f"[{pdf_file}] Q{i+1} future failed: {e}")
                    answer = f"Error: {e}"
                    duration = 0
                cleaned = clean_llm_response(questions[i], answer)
                page_answers[i] = cleaned
                logger_local.debug(f"[{pdf_file}] Page {page_num + 1} Q{i + 1} in {duration:.2f}s")

        all_answers.extend(page_answers)

    if skip_pdf:
        logger_local.warning(f"[{pdf_file}] Skipped due to page load errors.")
        return pdf_file, []

    return pdf_file, all_answers

def process_pdfs(pdf_files, model_path, max_size, all_questions_per_page, gpu_cycle, max_workers):
    """
    Process the list of pdf_files on GPUs as per gpu_cycle and max_workers.
    Returns:
     - results: list of tuples (pdf_file, answers)
     - failed_files: list of pdf_files that failed to process
    """
    results, failed_files = [], []
    logger.info(f"Launching {max_workers} worker process(es)")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf = {
            executor.submit(
                process_single_pdf,
                pdf_file,
                model_path,
                max_size,
                all_questions_per_page,
                next(gpu_cycle)
            ): pdf_file for pdf_file in pdf_files
        }

        for future in as_completed(future_to_pdf):
            pdf_file = future_to_pdf[future]
            try:
                pdf_file, answers = future.result()
                if not answers:
                    logger.warning(f"[Retry] {pdf_file} returned empty answers.")
                    failed_files.append(pdf_file)
                else:
                    results.append((pdf_file, answers))
                    logger.info(f"Finished processing {pdf_file}")
            except Exception as e:
                logger.exception(f"Error processing {pdf_file}: {e}")
                failed_files.append(pdf_file)

    return results, failed_files

def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python llama_11b.py <pdf_folder_path>")
        return

    input_folder = sys.argv[1]
    if not os.path.isdir(input_folder):
        logger.error(f"Error: Input path is not a directory: {input_folder}")
        return

    model_path = LLAMA_MODEL_DIR
    if not model_path or not Path(model_path).exists():
        logger.error(f"Model directory not found: {model_path}. Falling back to stub output.")
        _write_stub_results(Path(input_folder), Path.cwd() / "results.csv")
        return

    max_size = 560
    pdf_files = get_pdf_files_from_folder(input_folder)
    logger.info(f"[llama] Found {len(pdf_files)} PDF(s) under '{input_folder}'.")
    if not pdf_files:
        logger.warning("[llama] No PDFs found. Writing header-only CSV.")
        write_results_to_csv("results.csv", [], _headers())
        return

    num_gpus = torch.cuda.device_count()
    logger.info(f"Detected {num_gpus} GPU(s).")
    if num_gpus == 0:
        logger.error("No GPUs found. Cannot run the model. Writing stub results.")
        _write_stub_results(Path(input_folder), Path.cwd() / "results.csv")
        return

    # QUESTIONS
    page1_questions = [
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
    page2_questions = [
        "From the scanned document image above, determine whether there is any handwritten text inside the boxed area for item 12...",
        "From the scanned document image above, determine whether there is any handwritten text inside the boxed area for item 13...",
        "From the scanned document image above, determine whether there is any handwritten text inside the boxed area for item 14...",
        "From the scanned document image above, extract ONLY the handwritten text inside the boxed area for item 15...",
        "From the scanned document image above, extract ONLY the handwritten text inside the boxed area for item 16...",
        "From the scanned document image above, extract ONLY the handwritten text inside the boxed area for item 17...",
    ]
    page3_questions = [
        "From the scanned document image above, extract ONLY the handwritten tick marks inside the boxed area for item 18(a)...",
        "From the scanned document image above, extract ONLY the handwritten date inside the boxed area for item 18(b)...",
        "From the scanned document image above, extract ONLY the handwritten tick mark inside the boxed area for item 18(c)...",
    ]

    all_questions_per_page = [page1_questions, page2_questions, page3_questions]
    headers = _headers()

    gpu_cycle = itertools.cycle(range(num_gpus))
    max_workers = min(num_gpus, len(pdf_files))
    logger.info(f"Starting parallel PDF processing with max_workers={max_workers} ...")

    # First pass
    results, failed = process_pdfs(pdf_files, model_path, max_size, all_questions_per_page, gpu_cycle, max_workers)

    # Retry loop
    MAX_RETRIES = 3
    retry_count = 0
    while failed and retry_count < MAX_RETRIES:
        retry_count += 1
        logger.warning(f"Retry {retry_count}/{MAX_RETRIES} for {len(failed)} failed/skipped PDFs...")
        time.sleep(5)
        gpu_cycle = itertools.cycle(range(num_gpus))
        retry_results, failed = process_pdfs(failed, model_path, max_size, all_questions_per_page, gpu_cycle, max_workers)
        results.extend(retry_results)

    if failed:
        logger.error(f"The following PDFs still failed after {MAX_RETRIES} retries:")
        for fpdf in failed:
            logger.error(f" - {fpdf}")

    # Sort and write CSV
    results.sort(key=lambda x: x[0])
    total_questions = sum(len(qs) for qs in all_questions_per_page)
    rows = []
    for pdf_file, answers in results:
        if len(answers) < total_questions:
            answers += [""] * (total_questions - len(answers))
        rows.append([pdf_file] + answers)

    write_results_to_csv("results.csv", rows, headers)
    logger.info(f"[llama] Saved results.csv with {len(rows)} data row(s).")
    print_gpu_usage()  # logs info via logger

if __name__ == "__main__":
    main()