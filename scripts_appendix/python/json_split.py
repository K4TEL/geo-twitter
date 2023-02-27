import json
import os

# specify the input and output file names
folder = "datasets/"
input_file = 'worldwide-twitter-2021.jsonl'
output_file_prefix = f'{folder}worldwide-twitter-2021_'
max_lines_per_file = 20000000

# read the input file and split the data into multiple files
with open(f"{folder}{input_file}", 'r') as f:
    line_count = 0
    file_count = 0
    out_f = None
    for line in f:
        if line_count % max_lines_per_file == 0:
            # create a new output file
            if out_f:
                out_f.close()
            output_file_name = f'{output_file_prefix}{file_count}.jsonl'
            out_f = open(output_file_name, 'w')
            file_count += 1
        json_obj = json.loads(line.strip())
        out_f.write(f'{line.strip()}\n')
        line_count += 1
    if out_f:
        out_f.close()

