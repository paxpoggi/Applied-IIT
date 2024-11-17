import pandas as pd

def clean_and_pivot_csv(input_file, output_file):
    # Read the CSV file into a pandas DataFrame with Latin-1 encoding
    df = pd.read_csv(input_file, delimiter=',', encoding='latin-1')

    # Strip leading/trailing whitespaces from column names
    df.columns = df.columns.str.strip()

    # Print the columns present in the DataFrame to check if 'presentation_type' is there
    print(df.columns)

    # Filter out rows with "flashes" in the presentation_type column
    df = df[df['presentation_type'] != 'flashes']

    # Define the custom order of visual areas
    visual_area_order = ['LGd', 'VISp', 'VISl', 'VISrl', 'VISpm', 'VISam']

    # Reorder the 'visual_area' column based on the custom order
    df['visual_area'] = pd.Categorical(df['visual_area'], categories=visual_area_order, ordered=True)

    # Pivot the DataFrame to have each presentation type as its own column
    df_pivot = df.pivot_table(index=['probe_name', 'visual_area', 'interval'],
                              columns='presentation_type',
                              values='phi_value',
                              aggfunc='first')

    # Reset the index to turn the grouped columns back into regular columns
    df_pivot = df_pivot.reset_index()

    # Write the cleaned and pivoted DataFrame back to a new CSV file
    df_pivot.to_csv(output_file, sep='\t', index=False)

if __name__ == "__main__":
    # use data from Clean Data 2 folder, Clean Data 3 doesn't need the str.strip() line
    input_csv_file = "/Users/poggi/Documents/Maier Lab/IIT Notebooks/Cleaned Data 2/cleaned_combined_data_gamma.csv"
    output_csv_file = "/Users/poggi/Documents/Maier Lab/IIT Notebooks/Clean Data 4/jasp_ordered_combined_gamma_no_flash.csv"

    clean_and_pivot_csv(input_csv_file, output_csv_file)
    print(f'Cleaned and pivoted data saved to {output_csv_file}')
