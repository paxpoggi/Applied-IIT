import re
import ast
import pandas as pd
import os
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
    return encoding, confidence

file_path = '/Users/poggi/Documents/Maier Lab/IIT Notebooks/Cleaned Data/cleaned_combined_data_alpha.csv'  # Replace with the actual file path
encoding, confidence = detect_encoding(file_path)
print(f"Detected encoding: {encoding}, Confidence: {confidence}")

def clean_data(file_path, output_file_path):
    # Read the CSV file with the appropriate encoding and specify the column names
    column_names = ['probe_name', 'visual_area', 'presentation_type', 'interval', 'phi_value']
    df = pd.read_csv(file_path, encoding='utf-8', names=column_names, skiprows=1)

    # Clean the visual_area column
    df['visual_area'] = df['visual_area'].apply(lambda x: ast.literal_eval(x)[0].strip("[]'"))

    # Convert the phi_value column to float
    df['phi_value'] = df['phi_value'].astype(float)

    # Save the cleaned data to a CSV file
    df.to_csv(output_file_path, index=False)
    print(f"Cleaned data saved to '{output_file_path}'")

# Specify the input file path
file_path = '/Users/poggi/Documents/Maier Lab/IIT Notebooks/Cleaned Data/cleaned_combined_data_alpha.csv'

# Specify the output file path
output_file_path = '/Users/poggi/Documents/Maier Lab/IIT Notebooks/Cleaned Data 2/cleaned_combined_data_alpha.csv'

# Call the clean_data function with the file paths
clean_data(file_path, output_file_path)
