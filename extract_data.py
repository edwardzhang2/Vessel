#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import re
import os
from copy import deepcopy

# Canonical cargo types as per your list (uppercase to match synthetic data)
CARGO_TYPES = [
    'CONTAINER', 'BULK/WOODCHIP/CEMENT/ORE', 'TANKER_CRUDE_FUEL',
    'TANKER_LNG', 'CHEMICAL', 'GENERAL'
]

# Map full country names to abbreviations
FLAG_ABBREVIATIONS = {
    'LIBERIA': 'LIB',
    'HONG KONG': 'HKG',
    'PANAMA': 'PAN',
    'SINGAPORE': 'SGP',
    'CHINA': 'CHR',
    'SOUTH KOREA': 'KRS',
    'NORWAY': 'NOR',
    'SPAIN': 'SPN',
    'UNITED STATES': 'USA',
    'UNITED KINGDOM': 'UK',
    'FRANCE': 'FRA',
    'GERMANY': 'GER',
    'ITALY': 'ITA',
    'JAPAN': 'JPN',
}

# ---------- general cleaners ----------

def clean_value(val):
    """Generic trim: strip spaces, remove **surrounding**, and a single trailing period."""
    if not isinstance(val, str):
        return val
    v = val.strip()
    if v.startswith("**") and v.endswith("**") and len(v) >= 4:
        v = v[2:-2].strip()
    if v.endswith('.'):
        v = v[:-1].strip()
    return v

def parse_float_safe(ans):
    if not ans or not isinstance(ans, str):
        return None
    match = re.search(r"[-+]?\d*\.\d+|\d+", ans.replace(',', ''))
    if match:
        try:
            return float(match.group())
        except Exception:
            return None
    return None

def special_bool_field(ans):
    if not ans or not isinstance(ans, str):
        return False

    ans_clean = ans.strip().lower()
    if ans_clean == "yes":
        return True
    elif ans_clean == "no":
        return False

    empty = {'no', 'none', 'blank', 'n/a', 'na', '', 'n/a.', 'blank.', '**blank**', '**n/a**'}
    for word in ans_clean.split():
        if word in empty:
            return False

    return True

def normalize_cargo_type(raw_cargo):
    if not raw_cargo or not isinstance(raw_cargo, str):
        return 'CONTAINER'
    raw_cargo_upper = raw_cargo.upper()
    for ctype in CARGO_TYPES:
        if ctype in raw_cargo_upper:
            return ctype
    return 'CONTAINER'

def normalize_flag(raw_flag):
    if not raw_flag or not isinstance(raw_flag, str):
        return None
    flag_upper = raw_flag.strip().upper()
    if len(flag_upper) == 3 and flag_upper.isalpha():
        return flag_upper
    if flag_upper in FLAG_ABBREVIATIONS:
        return FLAG_ABBREVIATIONS[flag_upper]
    for country_name, abbrev in FLAG_ABBREVIATIONS.items():
        if country_name in flag_upper:
            return abbrev
    return flag_upper if flag_upper else None

# ---------- MD538 filtering & remap ----------

def _formtype_key(rows):
    """Return the actual key name used for FormType, if any."""
    if not rows:
        return None
    for k in rows[0].keys():
        if k.lower() == "formtype":
            return k
    return None

def _filter_md538_rows(rows):
    """Return only rows that are MD538 (based on 'FormType' column if present)."""
    if not rows:
        return rows
    form_key = _formtype_key(rows)
    if form_key is None:
        # Backward compatibility: treat all rows as MD538
        return rows
    md538 = []
    for r in rows:
        ft = str(r.get(form_key, "")).strip().upper()
        if ft.startswith("MD538") or ft == "":
            md538.append(r)
    return md538

PAGE_Q_RE = re.compile(r"^Page(\d+)_Q(\d+)$", re.IGNORECASE)

def remap_to_md538_headers(row):
    """
    Create a dict that:
      - Preserves metadata (File, FormType, Filename) when present
      - Adds MD538_P{page}_Q{q} for every Page{page}_Q{q} key
      - Preserves any other non-empty extra fields
    """
    out = {}

    # Preserve common metadata if present
    for meta_key in ("File", "FormType", "Filename"):
        if meta_key in row:
            out[meta_key] = clean_value(row.get(meta_key))

    # Rewrite PageX_QY -> MD538_PX_QY
    for k, v in row.items():
        if not k:
            continue
        m = PAGE_Q_RE.match(k)
        if m:
            page_num, q_num = m.group(1), m.group(2)
            new_key = f"MD538_P{page_num}_Q{q_num}"
            out[new_key] = clean_value(v)
        else:
            # Keep other fields (avoid overwriting preserved metadata)
            if k not in out and k not in ("",):
                out[k] = clean_value(v)

    return out

# ---------- classification (preserved original logic) ----------

def _get(row, key):
    val = row.get(key, "")
    if val is None:
        return None
    return val.strip() if isinstance(val, str) else val

def extract_classification_row(row):
    """
    Original classification logic:
      - Reads from the original Page1_Q*, Page2_Q*, Page3_Q* fields
      - Produces the 10 classification features used by classify.py
    """
    # Cargo Type and numeric fields
    cargo_type_raw = _get(row, 'Page1_Q8')
    cargo_type = normalize_cargo_type(cargo_type_raw)

    length = parse_float_safe(_get(row, 'Page1_Q4'))
    draft  = parse_float_safe(_get(row, 'Page1_Q5'))

    # National colors / flag
    flag_raw = _get(row, 'Page1_Q3')
    flag_raw_lower = flag_raw.lower() if flag_raw else ""
    matched_abbrev = None
    for country_name, abbrev in FLAG_ABBREVIATIONS.items():
        if country_name.lower() in flag_raw_lower:
            matched_abbrev = abbrev
            break
    flag = matched_abbrev if matched_abbrev else normalize_flag(flag_raw)

    # Hull details
    hull_raw = _get(row, 'Page3_Q1')
    single_hull = False
    double_sides = False
    double_bottoms = False
    if hull_raw:
        hull_raw_low = hull_raw.lower()
        if 'single hull' in hull_raw_low:
            single_hull = True
        if 'double sides' in hull_raw_low:
            double_sides = True
        if 'double bottoms' in hull_raw_low:
            double_bottoms = True
        if hull_raw_low.strip() in ['none', 'blank', 'n/a']:
            single_hull = False
            double_sides = False
            double_bottoms = False

    # Boolean handwritten flags (YES/NO presence)
    liquefied_gas   = special_bool_field(_get(row, 'Page2_Q1'))
    oil_over_2000t  = special_bool_field(_get(row, 'Page2_Q2'))
    noxious_liquid  = special_bool_field(_get(row, 'Page2_Q3'))

    data = {
        'Cargo Type': cargo_type,
        'Length Overall (m)': length,
        'Draft (m)': draft,
        'Flag': flag,
        'Single_Hull': single_hull,
        'Double_Sides': double_sides,
        'Double_Bottoms': double_bottoms,
        'Liquefied_Gas': liquefied_gas,
        'Oil_Over_2000t': oil_over_2000t,
        'Noxious_Liquid': noxious_liquid,
    }

    for k, v in data.items():
        if isinstance(v, str):
            data[k] = clean_value(v)

    return data

# ---------- MD537 in-place cleaning ----------

MD537_FIELD_PREFIX = "MD537_"

def _is_md537_row(row, form_key: str | None) -> bool:
    if form_key:
        ft = str(row.get(form_key, "")).strip().upper()
        if ft.startswith("MD537"):
            return True
    # Fallback: presence of any MD537_* field
    return any(k.startswith(MD537_FIELD_PREFIX) for k in row.keys())

def _clean_md537_value(key: str, val: str) -> str:
    """Apply MD537-specific cleaning rules to a single cell."""
    v = clean_value(val)
    if key == "MD537_P1_Q3":
        # Extract only the numeric part; keep as string (empty if none)
        m = re.search(r"[-+]?\d*\.?\d+", v.replace(',', '')) if isinstance(v, str) else None
        v = m.group() if m else ""
    return v

def clean_md537_rows_in_place(rows):
    """
    Return a new list of rows where MD537 rows have been cleaned:
      - Every MD537_* column gets trimmed, **stripped**, and trailing '.' removed
      - MD537_P1_Q3 extracts only the number
    """
    if not rows:
        return rows

    form_key = _formtype_key(rows)
    cleaned = []
    for r in rows:
        if _is_md537_row(r, form_key):
            nr = deepcopy(r)
            for k, v in r.items():
                if k.startswith(MD537_FIELD_PREFIX) and isinstance(v, str):
                    nr[k] = _clean_md537_value(k, v)
            cleaned.append(nr)
        else:
            cleaned.append(r)
    return cleaned

def _write_results_csv(path: str, rows: list[dict], original_fieldnames: list[str] | None):
    """
    Write rows back to results.csv. Keep original header order when possible,
    but include any new keys that may have appeared.
    """
    # Union of keys
    key_union = []
    def _add_key(k):
        if k not in key_union:
            key_union.append(k)

    if original_fieldnames:
        for k in original_fieldnames:
            _add_key(k)

    for r in rows:
        for k in r.keys():
            _add_key(k)

    # Write atomically
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=key_union)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    os.replace(tmp_path, path)

# ---------- main ----------

def main():
    input_csv = 'results.csv'
    output_csv = 'input.csv'

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"[extract] {input_csv} not found.")

    # Read results.csv
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        original_headers = reader.fieldnames[:] if reader.fieldnames else None
        all_rows = list(reader)

    print(f"[extract] Loaded results.csv with {len(all_rows)} total row(s).")

    # 1) Clean MD537 rows in-place and write back to results.csv
    cleaned_rows = clean_md537_rows_in_place(all_rows)
    if cleaned_rows != all_rows:
        _write_results_csv(input_csv, cleaned_rows, original_headers)
        print("[extract] MD537 rows cleaned in-place and results.csv updated.")
    else:
        print("[extract] No MD537 rows found to clean (results.csv unchanged).")

    # 2) Preserve existing behavior: filter MD538 rows and create input.csv for classification
    rows_for_md538 = _filter_md538_rows(cleaned_rows)
    print(f"[extract] Retained {len(rows_for_md538)} MD538 row(s) for classification.")

    # Build transformed rows that include:
    #   - MD538_P… remapped columns
    #   - original classification features (Cargo Type, etc.)
    transformed = []
    for r in rows_for_md538:
        md538_side = remap_to_md538_headers(r)     # adds MD538_Px_Qy + metadata
        cls_side   = extract_classification_row(r) # preserves your existing logic
        merged = {**md538_side, **cls_side}
        transformed.append(merged)

    # Build a stable header order for input.csv:
    #   1) Metadata first (if present)
    #   2) MD538_P{page}_Q{q} columns sorted by page then q
    #   3) Classification columns (fixed order)
    #   4) Any remaining extra fields in encounter order
    meta_order = [k for k in ("File", "FormType", "Filename") if any(k in r for r in transformed)]

    # Collect MD538 question keys
    md538_keys = set()
    extras_in_order = []
    for r in transformed:
        for k in r.keys():
            if k in meta_order:
                continue
            if k.startswith("MD538_P"):
                md538_keys.add(k)
            else:
                if k not in extras_in_order:
                    extras_in_order.append(k)

    def _pq_sort_key(key):
        m = re.match(r"^MD538_P(\d+)_Q(\d+)$", key)
        if not m:
            return (9999, 9999)
        return (int(m.group(1)), int(m.group(2)))

    md538_sorted = sorted(md538_keys, key=_pq_sort_key)

    classification_headers = [
        'Cargo Type', 'Length Overall (m)', 'Draft (m)', 'Flag',
        'Single_Hull', 'Double_Sides', 'Double_Bottoms',
        'Liquefied_Gas', 'Oil_Over_2000t', 'Noxious_Liquid'
    ]

    fieldnames = (
        meta_order
        + md538_sorted
        + [h for h in classification_headers if any(h in r for r in transformed)]
        + [k for k in extras_in_order if k not in md538_sorted and k not in classification_headers]
    )

    # Always write input.csv, even if empty (header only)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for r in transformed:
            writer.writerow(r)

    print(f"[extract] Saved '{output_csv}' with {len(transformed)} row(s).")

if __name__ == '__main__':
    main()
