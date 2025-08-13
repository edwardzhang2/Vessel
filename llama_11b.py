import os
import sys
import subprocess
import time
import torch

# Set PyTorch CUDA alloc config early to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import fitz
import itertools
import csv

def print_gpu_usage():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        print("GPU Usage:")
        for line in result.stdout.strip().split("\n"):
            index, util, mem_used, mem_total = line.split(", ")
            print(f"  GPU {index}: Utilization: {util}%  VRAM Used: {mem_used} MiB / {mem_total} MiB")
    except Exception as e:
        print(f"Warning: Could not get GPU usage info via nvidia-smi: {e}")

def ask_single_question_on_device(model, processor, image, question, device_id):
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
        end = time.time()
        duration = end - start

        output_text = processor.decode(outputs[0], skip_special_tokens=True)
        if output_text.startswith(input_text):
            output_text = output_text[len(input_text):].strip()
        pos = output_text.lower().find("assistant")
        if pos != -1:
            output_text = output_text[pos + len("assistant"):].strip()

        return question, output_text, duration

    except torch.cuda.OutOfMemoryError as oom:
        print(f"CUDA OOM error on question '{question}': {oom}")
        torch.cuda.empty_cache()
        return question, "Error: CUDA Out Of Memory", 0
    except Exception as e:
        print(f"Unexpected error on question '{question}': {e}")
        return question, f"Error: {e}", 0

def load_model_on_device(model_path, device_id):
    device = torch.device(f"cuda:{device_id}")
    print(f"Loading model on GPU {device_id}...")
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None
    )
    model.to(device)
    model.eval()
    print(f"Model loaded on GPU {device_id}.")
    return model

def pdf_page_to_image(pdf_path, page_number, max_size=560):
    doc = fitz.open(pdf_path)
    if page_number >= len(doc):
        raise ValueError(f"PDF '{pdf_path}' only has {len(doc)} pages.")
    page = doc[page_number]
    pix = page.get_pixmap(dpi=300)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
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

def get_pdf_files_from_folder(folder_path):
    supported = ('.pdf',)
    return [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.lower().endswith(supported)]

def clean_llm_response(question, raw_text):
    return raw_text.strip()

def process_single_pdf(pdf_file, model_path, max_size, all_questions_per_page, device_id):
    """
    Process one PDF file: load model on device, process each page and its questions in parallel threads,
    and return answers.
    """
    import torch
    import fitz
    from PIL import Image
    from transformers import MllamaForConditionalGeneration, AutoProcessor

    device = torch.device(f"cuda:{device_id}")
    processor = AutoProcessor.from_pretrained(model_path)

    print(f"[{pdf_file}] Loading model on GPU {device_id} in subprocess...")
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None
    )
    model.to(device)
    model.eval()
    print(f"[{pdf_file}] Model loaded on GPU {device_id}.")

    all_answers = []
    skip_pdf = False

    for page_num, questions in enumerate(all_questions_per_page):
        try:
            image = pdf_page_to_image(pdf_file, page_num, max_size)
        except Exception as e:
            print(f"[{pdf_file}] Failed to load page {page_num + 1}: {e}")
            skip_pdf = True
            break

        print(f"[{pdf_file}] Loaded page {page_num + 1}")

        page_answers = [None] * len(questions)

        with ThreadPoolExecutor(max_workers=len(questions)) as executor:
            futures = {}
            for i, question in enumerate(questions):
                future = executor.submit(
                    ask_single_question_on_device,
                    model, processor, image, question, device_id
                )
                futures[future] = i

            for future in as_completed(futures):
                i = futures[future]
                try:
                    question, answer, duration = future.result()
                except Exception as e:
                    answer = f"Error: {e}"
                    duration = 0

                cleaned = clean_llm_response(questions[i], answer)
                page_answers[i] = cleaned

                print(f"[{pdf_file}] Page {page_num + 1} Question {i + 1} answered in {duration:.2f}s")

        all_answers.extend(page_answers)

    if skip_pdf:
        print(f"[{pdf_file}] Skipped due to errors.")
        return pdf_file, []

    return pdf_file, all_answers

def process_pdfs(pdf_files, model_path, max_size, all_questions_per_page, gpu_cycle, max_workers):
    """
    Process the list of pdf_files on GPUs as per gpu_cycle and max_workers.
    Returns:
     - results: list of tuples (pdf_file, answers)
     - failed_files: list of pdf_files that failed to process
    """
    results = []
    failed_files = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf = {}
        for pdf_file in pdf_files:
            device_id = next(gpu_cycle)
            future = executor.submit(
                process_single_pdf,
                pdf_file,
                model_path,
                max_size,
                all_questions_per_page,
                device_id
            )
            future_to_pdf[future] = pdf_file

        for future in as_completed(future_to_pdf):
            pdf_file = future_to_pdf[future]
            try:
                pdf_file, answers = future.result()
                if len(answers) == 0:
                    print(f"[Retry] {pdf_file} returned empty answers, marking as failed.")
                    failed_files.append(pdf_file)
                else:
                    results.append((pdf_file, answers))
                    print(f"Finished processing {pdf_file}")
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                failed_files.append(pdf_file)

    return results, failed_files

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_folder_of_pdfs_parallel.py <pdf_folder_path>")
        return

    input_folder = sys.argv[1]

    if not os.path.isdir(input_folder):
        print("Error: Input path is not a directory.")
        return

    model_path = "./Llama-3.2-11B-Vision-Instruct"
    max_size = 560

    pdf_files = get_pdf_files_from_folder(input_folder)
    if not pdf_files:
        print("No PDF files found in the folder.")
        return

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found. Require at least 1 CUDA GPU.")
        return
    print(f"Number of GPUs available: {num_gpus}")

    # Updated prompts for items 12, 13, 14 to detect handwriting presence as YES or NO
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
        "From the scanned document image above, determine whether there is any handwritten text inside the boxed area for item 12, titled:' In the case of a vessel carrying liquefied gas in bulk, details of any certificate of fitness, including number, name of issuer, issue date, latest survey date, expiry date, and type of liquefied gas carried.' Do NOT extract or include the actual text content, nor any printed text or handwriting outside this box. If the box is blank or contains only 'N/A', output exactly 'NO'. If there is any handwritten content present inside the box, output exactly 'YES'. Do not output anything else.",
        "From the scanned document image above, determine whether there is any handwritten text inside the boxed area for item 13, titled: 'In the case of a vessel carrying more than 2000 tonnes of oil in bulk, details of any certificate of insurance against oil pollution, including certificate number, issuer, issue date, and expiry date.' Do NOT extract or include the actual text content, nor any printed text or handwriting outside this box. If the box is blank or contains only 'N/A', output exactly 'NO'. If there is any handwritten content present inside the box, output exactly 'YES'. Do not output anything else.",
        "From the scanned document image above, determine whether there is any handwritten text inside the boxed area for item 14, titled: 'In the case of a vessel carrying any noxious liquid substances in bulk, details of the International Pollution Prevention Certificate, including number, issuer, issue date, latest survey date, expiry date, and details if for loading, discharge, transshipment, or transit.' Do NOT extract or include the actual text content, nor any printed text or handwriting outside this box. If the box is blank or contains only 'N/A', output exactly 'NO'. If there is any handwritten content present inside the box, output exactly 'YES'. Do not output anything else.",
        "From the scanned document image above, extract ONLY the handwritten text inside the boxed area for item 15: 'Whether a Marpol surveyor is required (insert 'Yes' or 'No').' Do NOT include any printed text or handwriting from outside this box. If the box is empty or only contains 'N/A', return exactly 'N/A' or 'blank'. Output exclusively the handwritten content from this box. The handwriting should be at the right-side of the box. The answer should be in the format of 'Yes' or 'No'.",
        "From the scanned document image above, extract ONLY the handwritten text inside the boxed area for item 16: 'Whether a fixed inert gas system is fitted in the vessel (insert 'Yes' or 'No').' Do NOT include any printed text or handwriting from outside this box. If the box is empty or only contains 'N/A', return exactly 'N/A' or 'blank'. Output exclusively the handwritten content from this box. The handwriting should be at the right-side of the box. The answer should be in the format of 'Yes' or 'No'.",
        "From the scanned document image above, extract ONLY the handwritten text inside the boxed area for item 17: 'Whether a fixed tank washing system is fitted in the vessel (insert 'Yes' or 'No').' Do NOT include any printed text or handwriting from outside this box. If the box is empty or only contains 'N/A', return exactly 'N/A' or 'blank'. Output exclusively the handwritten content from this box. The handwriting should be at the right-side of the box. The answer should be in the format of 'Yes' or 'No'.",
    ]
    page3_questions = [
        "From the scanned document image above, extract ONLY the handwritten tick marks inside the boxed area for item 18(a), titled: 'Whether the vessel is single-hull, double sides, and/or double bottoms.' Identify which boxes are ticked (or checkmarked). Output ONLY the labels of the options ticked, separated by commas if multiple (e.g., 'double sides, double bottoms'). If none are ticked, output 'none'. Do NOT include any printed text or handwriting outside this box.",
        "From the scanned document image above, extract ONLY the handwritten date inside the boxed area for item 18(b), titled: 'The delivery date of the vessel (YY/MM/DD).' Output EXACTLY the handwritten date as it appears (e.g., '25/07/21'). If the box is empty or contains 'N/A', output 'blank' or 'N/A' accordingly. Do NOT include any printed text or handwriting outside this box.",
        "From the scanned document image above, extract ONLY the handwritten tick mark inside the boxed area for item 18(c), titled: 'Compliance with the Condition Assessment Scheme (CAS); and provision of Protective Location (PL) and Hydro-static Balance Loading (HBL). Tick if applicable.' If the box is ticked, output 'tick'. If the box is empty, output 'no tick'. Do NOT include any printed text or handwriting outside this box.",
    ]

    all_questions_per_page = [page1_questions, page2_questions, page3_questions]

    csv_headers = ['File'] + [f"Page{p + 1}_Q{i + 1}" for p, page_qs in enumerate(all_questions_per_page) for i in range(len(page_qs))]

    # Round robin GPU assignment iterator
    gpu_cycle = itertools.cycle(range(num_gpus))

    # Initial max_workers (concurrent processes)
    max_workers = min(num_gpus, len(pdf_files))

    print("Starting parallel PDF processing on available GPUs...")

    # Initial full batch processing
    results, failed = process_pdfs(pdf_files, model_path, max_size, all_questions_per_page, gpu_cycle, max_workers)

    MAX_RETRIES = 3
    retry_count = 0

    # Retry loop for failed/skipped PDFs
    while failed and retry_count < MAX_RETRIES:
        retry_count += 1
        print(f"\nRetry attempt {retry_count} for {len(failed)} failed/skipped PDFs...")

        # Optional delay to help with GPU memory cleanup
        time.sleep(5)

        # Reset GPU cycle for retry
        gpu_cycle = itertools.cycle(range(num_gpus))
        
        retry_results, failed = process_pdfs(failed, model_path, max_size, all_questions_per_page, gpu_cycle, max_workers)

        # Append retry results
        results.extend(retry_results)

    if failed:
        print(f"\nThe following PDFs still failed after {MAX_RETRIES} retries:")
        for fpdf in failed:
            print(f" - {fpdf}")

    # Sort results by filename
    results.sort(key=lambda x: x[0])

    # Prepare CSV rows, padding empty answers for failed files
    total_questions = sum(len(qs) for qs in all_questions_per_page)
    csv_rows = []
    for pdf_file, answers in results:
        if len(answers) < total_questions:
            answers += [""] * (total_questions - len(answers))
        csv_rows.append([pdf_file] + answers)

    write_results_to_csv("results.csv", csv_rows, csv_headers)
    print("\nAll PDFs processed. Results saved to 'results.csv'.")
    print_gpu_usage()

if __name__ == "__main__":
    main()