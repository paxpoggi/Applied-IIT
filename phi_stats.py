import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import seaborn as sns
import numpy as np
import re
import os
from sklearn.preprocessing import OneHotEncoder

def perform_linear_regression(data, output_file):
    # Group the data by visual area
    grouped_data = data.groupby('visual_area')

    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=['Visual Area', 'Presentation Type', 'Coefficient'])

    # Perform linear regression for each visual area
    for visual_area, group in grouped_data:
        # Get the x (presentation types) and y (phi values) values
        x = pd.get_dummies(group['presentation_type'], drop_first=True)
        y = group['phi_value']

        # Create a linear regression model
        model = LinearRegression()

        # Fit the model
        model.fit(x, y)

        # Store the results in the DataFrame
        for i, presentation_type in enumerate(x.columns):
            coefficient = model.coef_[i]
            results_df = pd.concat([
                results_df,
                pd.DataFrame([[visual_area, presentation_type, coefficient]], columns=results_df.columns)]
            )

    # Save the results to a CSV file
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to '{output_file}'")

# Read the CSV file into a DataFrame
data = pd.read_csv('/Users/poggi/Documents/Maier Lab/IIT Notebooks/Cleaned Data 2/cleaned_combined_data_gamma.csv')
output_file = '/Users/poggi/Documents/Maier Lab/IIT Notebooks/Phi Stats/linear_regression_results.csv'
#perform_linear_regression(data, output_file)


def create_average_phi_graph(input_folder, output_folder):
    # Get a list of all files in the input folder
    files = os.listdir(input_folder)

    # Create a dictionary to store the average phi values for each visual area and presentation type
    average_phi_values = {}

    # Loop over each file
    for file in files:
        # Construct the full path of the input file
        file_path = os.path.join(input_folder, file)

        # Read the CSV file with 'latin1' encoding
        df = pd.read_csv(file_path, encoding='latin1')

        # Print the column names in the DataFrame
        print(f"Column names in {file}: {df.columns}")

        # Validate if the 'phi_value' column exists in the DataFrame
        if 'phi_value' not in df.columns:
            print(f"Column 'phi_value' not found in {file}. Skipping this file.")
            continue

        # Group the data by visual area and presentation type, and calculate the average phi value
        grouped_data = df.groupby(['visual_area', 'presentation_type'])['phi_value'].mean().reset_index()
        # Separate out the rows with "flashes"
        flashes_data = grouped_data[grouped_data['presentation_type'] == 'flashes']

        # Filter out rows with "flashes" from the main DataFrame
        grouped_data = grouped_data[grouped_data['presentation_type'] != 'flashes']

        # Loop over each row in the grouped data
        for _, row in grouped_data.iterrows():
            visual_area = row['visual_area']
            presentation_type = row['presentation_type']
            phi_value = row['phi_value']

            # Add the phi value to the average_phi_values dictionary
            if visual_area not in average_phi_values:
                average_phi_values[visual_area] = {}

            if presentation_type not in average_phi_values[visual_area]:
                average_phi_values[visual_area][presentation_type] = []

            average_phi_values[visual_area][presentation_type].append(phi_value)

    # Plot the average phi values for each visual area and presentation type
    presentation_types = ['static_gratings', 'gabors', 'natural_scenes', 'natural_move_one', 'natural_move_three']
    x_ticks = range(len(presentation_types))

    for visual_area, values in average_phi_values.items():
        y_values = [sum(values.get(presentation_type, [0])) / len(values.get(presentation_type, [0])) for presentation_type in presentation_types]
        plt.plot(x_ticks, y_values, label=visual_area)

    presentation_labels = ['Static Gratings', 'Gabors', 'Natural Scenes', 'Natural Movie One', 'Natural Movie Three']

    plt.xticks(x_ticks, presentation_labels, rotation=0, ha='center', fontsize=8)
    plt.xlabel('Presentation Type')
    plt.ylabel('Average Phi Value')
    plt.legend()
    plt.title('Average Phi Values by Visual Area and Presentation Type')

    # adjust size of graph so that x ticks aren't cut off
    plt.gcf().subplots_adjust(bottom=0.3)

    # Save the graph as an image file
    output_file_path = os.path.join(output_folder, 'average_phi_graph_gamma_no_Flash_edited.png')
    plt.savefig(output_file_path)
    plt.close()

    print(f"Average phi graph saved to '{output_file_path}'")

input_folder = '/Users/poggi/Documents/Maier Lab/IIT Notebooks/Cleaned Data 2'
output_folder = '/Users/poggi/Documents/Maier Lab/IIT Notebooks/Phi Stats'
create_average_phi_graph(input_folder, output_folder)

def create_average_std_phi_graph(input_folder, output_folder):
    # Get a list of all files in the input folder
    files = os.listdir(input_folder)

    # Create a dictionary to store the average standard deviation of phi values for each visual area and presentation type
    average_std_phi_values = {}

    # Loop over each file
    for file in files:
        # Construct the full path of the input file
        file_path = os.path.join(input_folder, file)

        # Read the CSV file with 'latin1' encoding
        df = pd.read_csv(file_path, encoding='latin1')

        # Print the column names in the DataFrame
        print(f"Column names in {file}: {df.columns}")

        # Validate if the 'phi_value' column exists in the DataFrame
        if 'phi_value' not in df.columns:
            print(f"Column 'phi_value' not found in {file}. Skipping this file.")
            continue

        # Group the data by visual area and calculate the standard deviation of phi values
        grouped_data = df.groupby('visual_area')['phi_value'].std().reset_index()

        # Loop over each row in the grouped data
        for _, row in grouped_data.iterrows():
            visual_area = row['visual_area']
            std_phi_value = row['phi_value']

            # Add the standard deviation of phi values to the average_std_phi_values dictionary
            if visual_area not in average_std_phi_values:
                average_std_phi_values[visual_area] = []

            average_std_phi_values[visual_area].append(std_phi_value)

    # Plot the average standard deviation of phi values for each visual area as bar graphs
    visual_areas = list(average_std_phi_values.keys())
    x_ticks = range(len(visual_areas))
    bar_width = 0.35

    avg_std_values = [np.mean(average_std_phi_values[area]) for area in visual_areas]
    std_values = [np.std(average_std_phi_values[area]) for area in visual_areas]

    fig, ax = plt.subplots()

    ax.bar(x_ticks, avg_std_values, bar_width, yerr=std_values, label='Average Std Dev', align='center', alpha=0.75)

    plt.xticks(x_ticks, visual_areas, rotation=45, ha='right')
    plt.xlabel('Visual Area')
    plt.ylabel('Average Standard Deviation of Phi Value')
    plt.legend()
    plt.title('Average Standard Deviation of Phi Values by Visual Area Alpha Filtered')

    # adjust image size so that x-axis labels are not cut off
    fig.set_size_inches(10, 6)

    # Save the graph as an image file
    output_file_path = os.path.join(output_folder, 'average_std_phi_graph_alpha.png')
    plt.savefig(output_file_path)
    plt.close()

    print(f"Average standard deviation of phi graph saved to '{output_file_path}'")

input_folder = '/Users/poggi/Documents/Maier Lab/IIT Notebooks/Cleaned Data 2 Alpha'
output_folder = '/Users/poggi/Documents/Maier Lab/IIT Notebooks/Phi Stats'
# create_average_std_phi_graph(input_folder, output_folder)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_swarmplot(data_file, save_path=None, use_stripplot=True, marker_size=4):
    data = pd.read_csv(data_file)

    # Filter out 'flashes' data
    data_filtered = data[data['presentation_type'] != 'flashes']

    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed

    # Define the order in which the visual_area categories should appear on the x-axis
    visual_area_order = ['LGd', 'VISp', 'VISrl', 'VISl', 'VISpm', 'VISam']

    if use_stripplot:
        sns.stripplot(x='visual_area', y='phi_value', hue='presentation_type', data=data_filtered,
                      jitter=True, dodge=0.1, size=marker_size, alpha=0.7, order=visual_area_order)
    else:
        sns.swarmplot(x='visual_area', y='phi_value', hue='presentation_type', data=data_filtered,
                      size=marker_size, order=visual_area_order)

    plt.xlabel('Visual Area')
    plt.ylabel('Phi Value')
    plt.title('Phi Value vs. Visual Area')

    # Step 2: Update legend labels
    legend_labels = ['Static Gratings', 'Gabors', 'Natural Scenes', 'Natural Movie One', 'Natural Movie Three']
    legend = plt.legend(title='Presentation Type', labels=legend_labels, loc='center left', bbox_to_anchor=(.7, .83))

    # Step 4: Make the legend markers larger
    for handle in legend.legendHandles:
        handle.set_sizes([marker_size * 8])  # Adjust the factor as needed to make the markers larger

    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Swarmplot saved to: {save_path}")
    else:
        plt.show()

    plt.close()

#data_file_path = '/Users/poggi/Documents/Maier Lab/IIT Notebooks/Cleaned Data 2/cleaned_combined_data_gamma.csv'
#save_directory = '/Users/poggi/Documents/Maier Lab/IIT Notebooks/Phi Stats'

# Append the file name to the save directory
#save_path = os.path.join(save_directory, 'Swarmplot_poster_small.png')

# Call the function with stripplot option, smaller marker size, excluding flashes, and grouped by visual area
#create_swarmplot(data_file_path, save_path=save_path, use_stripplot=True, marker_size=2)


def create_swarmplot_overlap(data_file, save_path=None, use_stripplot=True, marker_size=2):
    data = pd.read_csv(data_file)

    # Filter out 'flashes' data
    data_filtered = data[data['presentation_type'] != 'flashes']

    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

    # Define the order in which the visual_area categories should appear on the x-axis
    visual_area_order = sorted(data_filtered['visual_area'].unique())

    if use_stripplot:
        # Create a dictionary to map visual area to numerical x-coordinates
        visual_area_mapping = {area: i for i, area in enumerate(visual_area_order)}

        # Map visual_area to numerical x-coordinates
        data_filtered['x_coordinate'] = data_filtered['visual_area'].map(visual_area_mapping)

        sns.stripplot(x='x_coordinate', y='phi_value', hue='presentation_type', data=data_filtered,
                      jitter=True, dodge=False, size=marker_size, alpha=0.7)

        # Set the x-axis ticks and labels
        plt.xticks(ticks=list(visual_area_mapping.values()), labels=list(visual_area_mapping.keys()))

    else:
        sns.swarmplot(x='visual_area', y='phi_value', hue='presentation_type', data=data_filtered,
                      size=marker_size, order=visual_area_order)

    plt.xlabel('Visual Area')
    plt.ylabel('Phi Value')
    plt.title('Swarmplot of Phi Value vs. Visual Area (Excluding Flashes)')

    plt.legend(title='Presentation Type', bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Swarmplot saved to: {save_path}")
    else:
        plt.show()

    plt.close()

#data_file_path = '/Users/poggi/Documents/Maier Lab/IIT Notebooks/Cleaned Data 2/cleaned_combined_data_gamma.csv'
#save_directory = '/Users/poggi/Documents/Maier Lab/IIT Notebooks/Phi Stats'

# Append the file name to the save directory
#save_path = os.path.join(save_directory, 'stripplot_grouped_by_visual_area_3.png')
# Call the function with stripplot option, smaller marker size, excluding flashes, and grouped by visual area
#create_swarmplot_overlap(data_file_path, save_path=save_path, use_stripplot=True, marker_size=3)





