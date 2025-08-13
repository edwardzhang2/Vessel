import csv

import re



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



def normalize_cargo_type(raw_cargo):

    """

    Normalize cargo type by matching against known cargo types (uppercase).

    Return canonical uppercase cargo type if matched, else default to 'CONTAINER'.

    """

    if not raw_cargo or not isinstance(raw_cargo, str):

        return 'CONTAINER'

    raw_cargo_upper = raw_cargo.upper()

    for ctype in CARGO_TYPES:

        if ctype in raw_cargo_upper:

            return ctype

    return 'CONTAINER'



def normalize_flag(raw_flag):

    """

    Normalize flag names to their 3-letter abbreviations.

    If already abbreviation (3 letters), return as uppercase.

    If full name matched, return abbreviation.

    Otherwise, return uppercased cleaned string or None.

    """

    if not raw_flag or not isinstance(raw_flag, str):

        return None

    flag_upper = raw_flag.strip().upper()

    # Return abbreviation if already 3-letter code

    if len(flag_upper) == 3 and flag_upper.isalpha():

        return flag_upper

    # Check for full country name mapping

    if flag_upper in FLAG_ABBREVIATIONS:

        return FLAG_ABBREVIATIONS[flag_upper]

    # Try substring match in mapping keys

    for country_name, abbrev in FLAG_ABBREVIATIONS.items():

        if country_name in flag_upper:

            return abbrev

    # Return cleaned uppercased string if no match found

    return flag_upper if flag_upper else None



def special_bool_field(ans):

    """

    Interprets the answer as presence or absence of handwritten text.

    Returns True if ans equals 'YES' (case-insensitive),

    Returns False if ans equals 'NO' or is in known empty words,

    else False by default.

    """

    if not ans or not isinstance(ans, str):

        return False



    ans_clean = ans.strip().lower()

    if ans_clean == "yes":

        return True

    elif ans_clean == "no":

        return False



    # Also treat known empty words as False

    empty = ['none', 'blank', 'n/a', 'na', '', 'n/a.', 'blank.', '**blank**', '**n/a**']

    if ans_clean in empty:

        return False



    # Default fallback

    return False



def parse_float_safe(ans):

    """

    Try to find the first float number in the input string,

    return it as float, or None if no number found.

    """

    if not ans or not isinstance(ans, str):

        return None

    # Regex to match a float number (integer or decimal)

    match = re.search(r"[-+]?\d*\.\d+|\d+", ans.replace(',', ''))

    if match:

        try:

            return float(match.group())

        except:

            return None

    return None



def clean_value(val):

    if not isinstance(val, str):

        return val

    val = val.strip()

    # Remove leading and trailing **

    if val.startswith("**") and val.endswith("**"):

        val = val[2:-2].strip()

    # Remove trailing '.' if present

    if val.endswith('.'):

        val = val[:-1].strip()

    return val



def extract_classification_row(row):

    def get(key):

        val = row.get(key, "")

        if val is None:

            return None

        return val.strip() if isinstance(val, str) else val



    # Cargo Type and numeric fields

    cargo_type_raw = get('Page1_Q8')

    cargo_type = normalize_cargo_type(cargo_type_raw)



    length = parse_float_safe(get('Page1_Q4'))

    draft = parse_float_safe(get('Page1_Q5'))



    # National colors - detect country name anywhere in text and map to abbreviation

    flag_raw = get('Page1_Q3')

    flag_raw_lower = flag_raw.lower() if flag_raw else ""



    matched_abbrev = None

    for country_name, abbrev in FLAG_ABBREVIATIONS.items():

        if country_name.lower() in flag_raw_lower:

            matched_abbrev = abbrev

            break



    if matched_abbrev:

        flag = matched_abbrev

    else:

        flag = normalize_flag(flag_raw)



    # Hull details

    hull_raw = get('Page3_Q1')

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



    # Boolean handwritten flags with updated YES/NO interpretation

    liquefied_gas = special_bool_field(get('Page2_Q1'))

    oil_over_2000t = special_bool_field(get('Page2_Q2'))

    noxious_liquid = special_bool_field(get('Page2_Q3'))



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



    # Clean string values in dictionary

    for k, v in data.items():

        if isinstance(v, str):

            data[k] = clean_value(v)



    return data



def main():

    input_csv = 'results.csv'

    output_csv = 'input.csv'



    with open(input_csv, 'r', encoding='utf-8') as f:

        reader = csv.DictReader(f)

        classification_rows = []

        for row in reader:

            classification_row = extract_classification_row(row)

            classification_rows.append(classification_row)



    classification_headers = [

        'Cargo Type', 'Length Overall (m)', 'Draft (m)', 'Flag',

        'Single_Hull', 'Double_Sides', 'Double_Bottoms',

        'Liquefied_Gas', 'Oil_Over_2000t', 'Noxious_Liquid'

    ]



    with open(output_csv, 'w', newline='', encoding='utf-8') as f_out:

        writer = csv.DictWriter(f_out, fieldnames=classification_headers)

        writer.writeheader()

        for row in classification_rows:

            writer.writerow(row)



    print(f"Classification data extracted and saved to '{output_csv}'.")



if __name__ == '__main__':

    main()