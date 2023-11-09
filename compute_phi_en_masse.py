import pandas as pd
import numpy as np
import pyphi
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from scipy.signal import butter, filtfilt
import random
import scipy.stats as stats
import os
import glob
import csv

# Disable PyPhi progress bars and welcome message
pyphi.config.PROGRESS_BARS = False
pyphi.config.WELCOME_OFF = True

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    y_abs = np.abs(y)
    return y_abs

def scatter_plot_phi_values(all_iterations_phi_values, probe_name, visual_area):
    # Data Preparation
    presentations = list(all_iterations_phi_values[0].keys())
    intervals = list(all_iterations_phi_values[0][presentations[0]].keys())

    # Initialize the data dict
    data = {presentation: [] for presentation in presentations}

    # Extract phi values for each presentation and interval from all iterations
    for presentation in presentations:
        for interval in intervals:
            phi_values = []
            for iteration in all_iterations_phi_values:
                phi_value = iteration[presentation][interval][0]
                phi_values.append(phi_value)
            data[presentation].extend(phi_values)

    # Create the graph
    fig, ax = plt.subplots()
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'yellow']

    # Determine the spacing between each presentation type
    spacing_between_presentations = 0.4  # Adjust this value as desired
    spacing_within_presentation = 0.2  # Adjust this value as desired

    # Offset for spacing between dots
    dot_offset = 0.05

    for i, presentation in enumerate(presentations):
        x_values = np.arange(len(intervals)) * spacing_within_presentation + i * (
                    len(intervals) * spacing_within_presentation + spacing_between_presentations)
        x_values = np.repeat(x_values, len(data[presentation]) // len(
            intervals))  # Repeat x_values to match the length of y_values
        x_values += dot_offset * (i - len(presentations) // 2)  # Add offset based on presentation index
        y_values = data[presentation]
        ax.scatter(x_values, y_values, label=presentation, color=colors[i])

    # Set x-axis labels and tick positions
    ax.set_xticks(np.arange(len(presentations)) * (
                    len(intervals) * spacing_within_presentation + spacing_between_presentations) + (
                                    len(intervals) - 1) * spacing_within_presentation / 2)
    ax.set_xticklabels(presentations, rotation=45)

    # Set axis labels and title
    ax.set_xlabel('Presentation Type (Duration 200s)')
    ax.set_ylabel('Phi-values')
    ax.set_title(f'Alpha Filtered Phi-values for Different Presentations in {visual_area} for {probe_name}')

    # Set legend with smaller font size
    legend = ax.legend(prop={'size': 8})

    # Save the figure with proper padding
    plt.savefig(f'phi_values_rand_alpha_{visual_area} for {probe_name}.png', bbox_inches='tight')

    # Display the graph
    plt.show()

def anova_analysis(all_iterations_phi_values):
    # Extract phi values for each presentation from all iterations
    data = {}

    for presentation in all_iterations_phi_values[0]:
        if presentation not in data:
            data[presentation] = []
        for interval in all_iterations_phi_values[0][presentation]:
            for iteration in all_iterations_phi_values:
                phi_value = iteration[presentation][interval][0]
                data[presentation].append(phi_value)

    # Perform one-way ANOVA
    fvalue, pvalue = stats.f_oneway(*data.values())

    # Create a pandas DataFrame to display the ANOVA results
    anova_results = pd.DataFrame({
        'Presentation': list(data.keys()),
        'Phi Values': list(data.values()),
    })

    # Add the ANOVA summary row
    anova_results.loc['ANOVA'] = ['', '']  # Empty row
    anova_results.loc['ANOVA', 'Presentation'] = 'Overall'
    anova_results.loc['ANOVA', 'Phi Values'] = [fvalue], [pvalue]  # Convert to lists

    # Set precision to 3 decimal points
    anova_results = anova_results.round(decimals={'Phi Values': 3})

    # Save the ANOVA results as an HTML file
    anova_html = anova_results.to_html(float_format=lambda x: '{:.3f}'.format(x))
    with open(f'anova_results_gamma_{visual_area}_{probe_name}.html', 'w') as file:
        file.write(anova_html)

    # Display the ANOVA results table
    print("One-way ANOVA results:")
    print(anova_results)

def compute_phi_lfp_rand_int(presentation_df, lfp_df, interval, channels, sampling_rate, presentation_name):
    assert isinstance(presentation_df, pd.DataFrame), f"presentation_df is {type(presentation_df)}, not a DataFrame"
    # Apply bandpass filter to each of the random channels separately
    lfp_df[channels] = lfp_df[channels].apply(
        lambda x: bandpass_filter(x, lowcut=35, highcut=65, fs=sampling_rate))

    if lfp_df.isnull().sum().sum() > 0:
        print("NaNs introduced after bandpass_filter")
    # Calculate the median values down each column for the random channels
    median_values = lfp_df.loc[:, channels].median(axis=0)

    if median_values.isnull().sum() > 0:
        print("NaNs introduced in median_values")

    for i in range(len(presentation_df)):
        start_time = presentation_df.start_time[i] + interval[0]
        end_time = presentation_df.start_time[i] + interval[1]

    # Calculate the start and end indices based on the sampling rate
    start_index = int(start_time * sampling_rate)
    end_index = int(end_time * sampling_rate)
    num_rows = end_index - start_index + 1

    # Create a new DataFrame to store the binary values within the desired time range
    binary_df = pd.DataFrame(index=range(num_rows), columns=channels)

    # Iterate over the columns (channels) of the random channels
    for column in channels:
        values = lfp_df[column][start_index:end_index + 1]
        if len(values) != len(binary_df):
            print(f"Size mismatch: binary_df has {len(binary_df)} rows, values from lfp_df has {len(values)} rows")
        # Iterate over the rows (data points) within the desired time range
        for i, value in enumerate(lfp_df[column][start_index:end_index + 1]):
            # Check if the value is above or below the average for that channel
            if value >= median_values[column]:
                binary_df.loc[i, column] = 1
            else:
                binary_df.loc[i, column] = 0

    allData = binary_df.to_numpy()
    print(allData.dtype)
    allData = allData.astype(np.int16)

    # get all possible unique states from the data
    possible_states = np.unique(allData, axis=0)
    print(possible_states.shape)

    if possible_states.shape != (8, 3):
        print("Skipping interval - incorrect shape of possible_states")
        return None, None

    # print all possible states
    for i in range(possible_states.shape[0]):
        print(possible_states[i, :])

    tpm_state_by_state = np.zeros((possible_states.shape[0], possible_states.shape[0]))

    # loop over each possible state at time t
    for i in range(possible_states.shape[0]):
        state_t = possible_states[i, :]

        # find the indices where state_t appears in allData
        idx_t = np.all(allData == state_t, axis=1)

        # find the indices where state_t+1 appears in allData
        idx_t_plus_1 = np.where(idx_t)[0] + 1

        # exclude the last index to prevent out of bounds error
        if np.any(idx_t_plus_1 >= allData.shape[0]):
            idx_t_plus_1 = idx_t_plus_1[:-1]

        # get the state at time t+1 for each occurrence of state_t
        states_t_plus_1 = allData[idx_t_plus_1, :]

        # loop over each possible state at time t+1
        for j in range(possible_states.shape[0]):
            state_t_plus_1 = possible_states[j, :]

            # find the indices where state_t+1 matches state_t_plus_1
            idx_t_plus_1_matching = np.all(states_t_plus_1 == state_t_plus_1, axis=1)

            # calculate the transition probability from state_t to state_t_plus_1
            tpm_state_by_state[i, j] = np.sum(idx_t_plus_1_matching) / np.sum(idx_t)

    # convert state by state TPM to state by node TPM
    sbn_tpm = pyphi.convert.state_by_state2state_by_node(tpm_state_by_state)
    # sbn_tpm = pyphi.convert.state_by_state2state_by_node(tpm_state_by_state_random)
    print(sbn_tpm.shape)
    # convert state by node TPM to state by state TPM to make it conditionally independent
    sbs_tpm = pyphi.convert.state_by_node2state_by_state(sbn_tpm)
    print(sbs_tpm.shape)
    # Create a unique filename based on presentation and interval
    output_folder = '/Users/poggi/Documents/Maier Lab/Figures (Pax)/TPMs'
    filename = os.path.join(output_folder,"tpm_{probe_name}_{presentation_name}_{interval[0]}_{interval[1]}.csv")
    df = pd.DataFrame(sbs_tpm)
    csv_data = df.to_csv(filename, index=False)
    # use state by state TPM as Phi Input
    pyPhiInput = sbs_tpm
    labels = ('A', 'B', 'C')
    # create network
    network = pyphi.Network(pyPhiInput, node_labels=labels)
    node_indices = (0, 1, 2)
    states = possible_states

    phi_values = np.zeros(len(states))
    max_phi = -np.inf
    max_phi_state = None

    # Loop through each possible state
    for i, state in enumerate(states):
        # Create the candidate subsystem
        candidate_system = pyphi.Subsystem(network, state, node_indices)

        # Compute the big phi value for the candidate subsystem
        phi_structure = pyphi.new_big_phi.phi_structure(candidate_system)
        big_phi = phi_structure.big_phi

        # Store the big phi value in the array
        phi_values[i] = big_phi

        # Update the maximum big phi value and its corresponding state if necessary
        if big_phi > max_phi:
            max_phi = big_phi
            max_phi_state = state

    return max_phi, max_phi_state


# Main script execution
if __name__ == '__main__':

    # Directory containing the data
    data_dir = '/Volumes/Extreme SSD/Phi_Test'

    # Collect the master file paths
    nwb_master_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*_ses-*.nwb')))
    probe_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*_ses-*_probe-*_ecephys.nwb')))

    # Exclude probe files from master files
    nwb_master_files = list(set(nwb_master_files) - set(probe_files))

    probes_lfp_df = []

    # Iterate over each master file
    for nwb_master_file in nwb_master_files:
        # Extract subject and session number from the master filenames
        filename_master = os.path.basename(nwb_master_file)
        filename_parts_master = filename_master.split('_')
        subject_num_master = filename_parts_master[0][4:]  # Remove 'sub-' prefix
        session_num_master = filename_parts_master[1].split('.')[0][4:]  # Remove 'ses-' prefix and '.nwb' suffix

        # Collect the corresponding nwb files
        nwb_files = sorted(
            glob.glob(
                os.path.join(data_dir, f'sub-{subject_num_master}_ses-{session_num_master}_probe-*_ecephys.nwb')))

        # Load the NWB Master file
        io_master = NWBHDF5IO(nwb_master_file, 'r', load_namespaces=True)
        nwb_master = io_master.read()

        # Check keys in intervals
        if 'flashes_presentations' not in nwb_master.intervals.keys():
            print(f"File '{nwb_master_file}' does not contain 'flashes_presentations'.")
            continue

        # Grab all the stimulus presentation data
        intervals_dict = nwb_master.intervals
        flashes_presentations = intervals_dict['flashes_presentations'].to_dataframe()
        static_gratings_presentations = intervals_dict['static_gratings_presentations'].to_dataframe()
        gabors_presentations = intervals_dict['gabors_presentations'].to_dataframe()
        natural_scenes_presentations = intervals_dict['natural_scenes_presentations'].to_dataframe()
        natural_move_one_presentations = intervals_dict['natural_movie_one_presentations'].to_dataframe()
        natural_move_three_presentations = intervals_dict['natural_movie_three_presentations'].to_dataframe()

        for nwb_file_path in nwb_files:
            # Load the ecephys probe file
            io_probe = NWBHDF5IO(nwb_file_path, 'r', load_namespaces=True)
            nwb = io_probe.read()

            # Extract the probe name from the file name
            probe_name = os.path.basename(nwb_file_path).split('_')[2].replace('-', '_')
            probe_data_key = f'{probe_name}_lfp_data'

            # Define desired brain areas
            desired_areas = ['LGd', 'VISp', 'VISpm', 'VISl', 'VISrl', 'VISam']
            # Determine indices of```python

            desired_indices = [i for i, loc in enumerate(nwb.electrodes['location']) if loc in desired_areas]

            # Determine the actual visual area for each probe
            actual_visual_areas = []
            for i in desired_indices:
                location = nwb.electrodes['location'][i]
                if location in desired_areas:
                    actual_visual_areas.append(location)
                    break

            # determine the group name
            group_name = nwb.electrodes['group_name'][0]

            sampling_rate = nwb.electrode_groups[group_name].lfp_sampling_rate

            for actual_visual_area in actual_visual_areas:
                lfp_data = nwb.acquisition
                lfp_data = lfp_data[probe_data_key]
                lfp_data = lfp_data.data
                lfp_df = pd.DataFrame(lfp_data)

                # Filter the DataFrame to include only desired columns
                lfp_df = lfp_df[desired_indices]

                # Check if the data is mostly zeros
                if not lfp_df.astype(bool).sum().sum() > 10000:
                    # Print a message indicating which probe has been removed
                    print(
                        f"Probe {probe_name} with visual area {actual_visual_areas} has been removed due to mostly zero data.")
                else:
                    # Append the DataFrame to the list
                    probes_lfp_df.append({
                        'probe_name': probe_name,
                        'visual_area': actual_visual_areas,
                        'lfp_data': lfp_df,
                        'sampling_rate': sampling_rate
                    })

            # Close the ecephys probe file
            io_probe.close()

        for i, probe_dict in enumerate(probes_lfp_df):
            lfp_data = probe_dict['lfp_data']
            if lfp_data.isna().any().any():
                # Replace NaN values with 0
                lfp_data = lfp_data.fillna(0)
                print(f"DataFrame {i} contains NaN values. Replacing with 0.")
            else:
                print(f"DataFrame {i} does not contain any NaN values")

        # Close the NWB Master file

        io_master.close()

    # Store all Phi values across all iterations for each probe
    all_probes_iterations_phi_values = []

    all_probes_data = {}

    presentations = [flashes_presentations, static_gratings_presentations, gabors_presentations,
                    natural_scenes_presentations, natural_move_one_presentations,
                    natural_move_three_presentations]

    presentation_names = ['flashes', 'static_gratings', 'gabors', 'natural_scenes', 'natural_move_one',
                        'natural_move_three']

    intervals = [(1, 50), (51, 100), (101, 150), (151, 200)]  # Define the intervals here

    #Iterate over each probe's lfp_df
    for lfp_data in probes_lfp_df:
        lfp_df = lfp_data['lfp_data']
        probe_name = lfp_data['probe_name']
        visual_area = lfp_data['visual_area']
        sampling_rate = lfp_data['sampling_rate']
        # Store all Phi values across all iterations for this probe
        all_iterations_phi_values = []

        for iter_count in range(4):
            # Choose three random channels
            random_channels = random.sample(list(lfp_df.columns), 3)

            # Store max phi and corresponding state for each presentation and interval
            presentation_phi_values = {}

            for i, presentation in enumerate(presentations):
                # Store max phi and corresponding state for each interval
                interval_phi_values = {}
                for presentation_name in presentation_names:
                    for interval in intervals:
                        max_phi, max_phi_state = compute_phi_lfp_rand_int(presentation, lfp_df, interval,
                                                                  random_channels, sampling_rate, presentation_name)
                        interval_phi_values[interval] = (max_phi, max_phi_state)
                        print(
                            f'Finished computing Phi value for {presentation_names[i]} - Interval {interval[0]}-{interval[1]} seconds')

            presentation_phi_values[presentation_names[i]] = interval_phi_values

            for presentation, intervals in presentation_phi_values.items():
                print(f'Presentation: {presentation}')
                for interval, phi_values in intervals.items():
                    print(f'Interval: {interval[0]}-{interval[1]} seconds')
                    print(f'Max Phi: {phi_values[0]}, Max Phi State: {phi_values[1]}')

            # Store the Phi values from this iteration
            all_iterations_phi_values.append(presentation_phi_values)

        # Add the phi values for this probe to the overall list
        all_probes_iterations_phi_values.append(all_iterations_phi_values)

        # anova_analysis(all_iterations_phi_values)

        scatter_plot_phi_values(all_iterations_phi_values, probe_name, visual_area)

        probe_data = {'probe_name': probe_name, 'visual_area': visual_area, 'phi_values': all_iterations_phi_values}

        all_probes_data[probe_name] = probe_data

        # Flatten the nested dictionary structure
        flattened_data = []
        for probe_name, probe_data in all_probes_data.items():
            for iteration_phi_values in probe_data['phi_values']:
                flattened_data.append({
                    'probe_name': probe_data['probe_name'],
                    'visual_area': probe_data['visual_area'],
                    'iteration_phi_values': iteration_phi_values
                })

        # Save as CSV
        fieldnames = ['probe_name', 'visual_area', 'iteration_phi_values']
        with open(f'all_probes_data_sub_{filename_master}_gamma_visualization.csv', 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened_data)








