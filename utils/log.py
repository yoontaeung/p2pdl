import json
import os

def save_results(result_data, result_file):
    # Ensure the result file exists and is not empty
    if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
        with open(result_file, 'r') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                # Handle the case where the file is not properly formatted JSON
                results = []
    else:
        results = []

    # Append new data
    results.append(result_data)

    # Write the updated results back to the file
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)