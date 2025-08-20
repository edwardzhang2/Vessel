#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Llama 3.2 11B Vision — MD537 + MD538 extractor (single-worker).
- Unified results.csv: FormType, File, then MD537_* and MD538_* column groups.
- Robust form detection:
  (1) page count heuristic (MD537:2, MD538:>=3)
  (2) confirm by LLM on bottom-left cropped region for page 1
- Local-only load (set LLAMA_MODEL_DIR, /cache).
"""

import os
import sys
import csv
import time
import subprocess
import logging
from pathlib import Path

LOG_DIR = Path(os.environ.get("LLAMA_LOG_DIR", "/tmp"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"llama_11b_{os.getpid()}.log"
LOG_LEVEL = os.environ.get("LLAMA_LOG_LEVEL", "INFO").upper()

root_logger = logging.getLogger()
if root_logger.handlers:
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] [pid=%(process)d] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE, mode="w")],
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to {LOG_FILE}")

# ---- STUB ----
def _headers_unified():
    md537_p1 = [f"MD537_P1_Q{i}" for i in range(1, 12)]  # 11
    md537_p2 = [f"MD537_P2_Q{i}" for i in range(1, 4)]   # 3
    md538_p1 = [f"MD538_P1_Q{i}" for i in range(1, 12)]  # 11
    md538_p2 = [f"MD538_P2_Q{i}" for i in range(1, 7)]   # 6
    md538_p3 = [f"MD538_P3_Q{i}" for i in range(1, 4)]   # 3
    return ["FormType", "File"] + md537_p1 + md537_p2 + md538_p1 + md538_p2 + md538_p3

def _write_stub_results(input_folder: Path, out_path: Path):
    pdfs = sorted(input_folder.rglob("*.pdf"))
    hdrs = _headers_unified()
    logger.warning(f"[STUB] Writing stub results for {len(pdfs)} PDF(s) to {out_path}")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hdrs)
        for pdf in pdfs:
            w.writerow(["MD538", str(pdf)] + [""] * (len(hdrs) - 2))

if os.environ.get("SKIP_LLAMA", "0") == "1":
    if len(sys.argv) < 2:
        logger.error("Usage: python llama_11b.py <pdf_folder_path>")
        sys.exit(0)
    in_dir = Path(sys.argv[1]).resolve()
    if not in_dir.is_dir():
        logger.error(f"Error: '{in_dir}' is not a directory.")
        sys.exit(0)
    out_file = Path.cwd() / "results.csv"
    _write_stub_results(in_dir, out_file)
    logger.info("[STUB] Done.")
    sys.exit(0)

# ---- heavy deps ----
import torch
from PIL import Image
import fitz  # PyMuPDF
from transformers import AutoModelForVision2Seq, AutoProcessor

os.environ.setdefault("HF_HOME", "/cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/cache")
LLAMA_MODEL_DIR = os.environ.get("LLAMA_MODEL_DIR", "/models/Llama-3.2-11B-Vision-Instruct")

# ---- helpers ----
def get_pdfs(folder_path: str):
    pdfs = []
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, fname))
    return sorted(pdfs)

def pdf_page_to_image(pdf_path, page_number, max_size=560):
    doc = fitz.open(pdf_path)
    try:
        if page_number >= len(doc):
            raise ValueError(f"PDF '{pdf_path}' only has {len(doc)} pages.")
        page = doc[page_number]
        pix = page.get_pixmap(dpi=300)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        doc.close()
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        image = image.resize((int(image.size[0]*ratio), int(image.size[1]*ratio)))
    return image

def crop_bottom_left(img: Image.Image, width_frac=0.5, height_frac=0.3) -> Image.Image:
    """Crop bottom-left region where 'MD537/MD538 (Rev ...)' lives."""
    w, h = img.size
    left = 0
    right = int(w * max(0.2, min(1.0, width_frac)))
    top = int(h * max(0.0, min(1.0, 1.0 - height_frac)))
    bottom = h
    return img.crop((left, top, right, bottom))

def ask_on_image(model, processor, image, prompt, device, max_new_tokens=60):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=image, text=input_text, return_tensors="pt", padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
    text = processor.decode(out[0], skip_special_tokens=True)
    if text.startswith(input_text):
        text = text[len(input_text):].strip()
    pos = text.lower().find("assistant")
    if pos != -1:
        text = text[pos+len("assistant"):].strip()
    return text.strip()

def detect_formtype_via_llm(model, processor, full_img, device):
    # Try on cropped corner first
    cropped = crop_bottom_left(full_img, width_frac=0.45, height_frac=0.28)
    strict = ("Look at this bottom-left corner of the form only. "
              "Answer exactly one token: MD537 or MD538.")
    ans = ask_on_image(model, processor, cropped, strict, device).upper()
    if "537" in ans:
        return "MD537"
    if "538" in ans:
        return "MD538"
    # Fallback: ask on the whole page
    whole = ("Look at the bottom-left corner of this page. "
             "Answer exactly one token: MD537 or MD538.")
    ans2 = ask_on_image(model, processor, full_img, whole, device).upper()
    if "537" in ans2:
        return "MD537"
    if "538" in ans2:
        return "MD538"
    return None

def decide_formtype(pdf_path, model, processor, device):
    """Combine page-count heuristic with cropped-LLM read."""
    # Heuristic by page count
    try:
        doc = fitz.open(pdf_path)
        n_pages = len(doc)
        doc.close()
    except Exception as e:
        logger.warning(f"[detect] Could not open {pdf_path} for page count: {e}")
        n_pages = None

    # Render page 1 once for LLM read
    try:
        img0 = pdf_page_to_image(pdf_path, 0, max_size=900)
    except Exception as e:
        logger.exception(f"[detect] Failed to render page 1 for {pdf_path}: {e}")
        # fall back to page count only
        if n_pages is not None and n_pages <= 2:
            return "MD537"
        return "MD538"

    llm_guess = detect_formtype_via_llm(model, processor, img0, device)
    count_guess = None
    if n_pages is not None:
        count_guess = "MD537" if n_pages <= 2 else "MD538"

    # Decision:
    if llm_guess and count_guess:
        if llm_guess != count_guess:
            logger.warning(f"[detect] Disagreement for {pdf_path}: LLM={llm_guess}, pages={count_guess}. Using LLM.")
        return llm_guess
    return llm_guess or count_guess or "MD538"

# Question sets
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
    "From the scanned document image above, determine whether there is any handwritten text inside the boxed area for item 12...",
    "From the scanned document image above, determine whether there is any handwritten text inside the boxed area for item 13...",
    "From the scanned document image above, determine whether there is any handwritten text inside the boxed area for item 14...",
    "From the scanned document image above, extract ONLY the handwritten text inside the boxed area for item 15...",
    "From the scanned document image above, extract ONLY the handwritten text inside the boxed area for item 16...",
    "From the scanned document image above, extract ONLY the handwritten text inside the boxed area for item 17...",
]
MD538_P3 = [
    "From the scanned document image above, extract ONLY the handwritten tick marks inside the boxed area for item 18(a)...",
    "From the scanned document image above, extract ONLY the handwritten date inside the boxed area for item 18(b)...",
    "From the scanned document image above, extract ONLY the handwritten tick mark inside the boxed area for item 18(c)...",
]

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

    pdf_files = get_pdfs(input_folder)
    logger.info(f"Found {len(pdf_files)} PDF(s).")
    if not pdf_files:
        with (Path.cwd() / "results.csv").open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_headers_unified())
        logger.info("No PDFs; wrote header-only results.csv")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else None,
            local_files_only=True,
        )
        model.to(device)
        model.eval()
    except Exception as e:
        logger.exception(f"Failed to load model/processor from '{model_path}': {e}")
        _write_stub_results(Path(input_folder), Path.cwd() / "results.csv")
        return

    headers = _headers_unified()
    out_path = Path.cwd() / "results.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for pdf in pdf_files:
            logger.info(f"Processing: {pdf}")

            # Decide form robustly (page count + LLm corner read)
            formtype = decide_formtype(pdf, model, processor, device)
            logger.info(f"[detect] {pdf} => {formtype}")

            # Prepare blank row (all columns except first two)
            row_data = [""] * (len(headers) - 2)

            def fill(prefix, answers):
                blocks = {
                    "MD537_P1": (headers.index("MD537_P1_Q1") - 2, len(MD537_P1)),
                    "MD537_P2": (headers.index("MD537_P2_Q1") - 2, len(MD537_P2)),
                    "MD538_P1": (headers.index("MD538_P1_Q1") - 2, len(MD538_P1)),
                    "MD538_P2": (headers.index("MD538_P2_Q1") - 2, len(MD538_P2)),
                    "MD538_P3": (headers.index("MD538_P3_Q1") - 2, len(MD538_P3)),
                }
                start, count = blocks[prefix]
                for i in range(min(count, len(answers))):
                    row_data[start + i] = answers[i] if answers[i] is not None else ""

            def ask_page_questions(page_index, questions):
                try:
                    img = pdf_page_to_image(pdf, page_index, max_size=700)
                except Exception as e:
                    logger.exception(f"Failed to render page {page_index+1} for {pdf}: {e}")
                    return [""] * len(questions)
                answers = []
                for q in questions:
                    try:
                        ans = ask_on_image(model, processor, img, q, device)
                    except Exception as e:
                        logger.exception(f"Q fail on page {page_index+1}: {e}")
                        ans = ""
                    answers.append(ans.strip())
                return answers

            if formtype == "MD537":
                a1 = ask_page_questions(0, MD537_P1)
                a2 = ask_page_questions(1, MD537_P2)
                fill("MD537_P1", a1)
                fill("MD537_P2", a2)
            else:
                a1 = ask_page_questions(0, MD538_P1)
                a2 = ask_page_questions(1, MD538_P2)
                a3 = ask_page_questions(2, MD538_P3)
                fill("MD538_P1", a1)
                fill("MD538_P2", a2)
                fill("MD538_P3", a3)

            writer.writerow([formtype, pdf] + row_data)

    logger.info(f"Wrote unified results.csv to: {out_path}")

    # Optional GPU stats
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        for line in res.stdout.strip().splitlines():
            logger.info(f"GPU: {line}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
