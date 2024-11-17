import pandas as pd
import numpy as np
import pyphi
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from tqdm import tqdm
from scipy.signal import butter, filtfilt
import random

# Disable PyPhi progress bars and welcome message
pyphi.config.PROGRESS_BARS = False
pyphi.config.WELCOME_OFF = True

# Loading nwb_master file
filepath = '/Users/poggi/Documents/Maier Lab/NWB Data/sub-699733573_ses-715093703.nwb'
io = NWBHDF5IO(filepath, 'r', load_namespaces=True)
nwb_master = io.read()  # nwb dataset

# Opening ecephys probe file
filepath = '/Users/poggi/Documents/Maier Lab/NWB Data/sub-699733573_ses-715093703_probe-810755797_ecephys.nwb'
io = NWBHDF5IO(filepath, 'r', load_namespaces=True)
nwb = io.read()  # nwb dataset

# Grab all the stimulus presentation data
intervals_dict = nwb_master.intervals
flashes_presentations = intervals_dict['flashes_presentations'].to_dataframe()
static_gratings_presentations = intervals_dict['static_gratings_presentations'].to_dataframe()
gabors_presentations = intervals_dict['gabors_presentations'].to_dataframe()
natural_scenes_presentations = intervals_dict['natural_scenes_presentations'].to_dataframe()
natural_move_one_presentations = intervals_dict['natural_movie_one_presentations'].to_dataframe()
natural_move_three_presentations = intervals_dict['natural_movie_three_presentations'].to_dataframe()

lfp_data = nwb.acquisition
lfp_data = lfp_data['probe_810755797_lfp_data']
lfp_data = lfp_data.data
lfp_df = pd.DataFrame(lfp_data)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter_gamma(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    y_abs = np.abs(y)
    return y_abs

def compute_phi_lfp_rand_int(presentation_df, lfp_df, nwb, interval, channels):
    # Retrieve the sampling rate from the NWB file
    sampling_rate = nwb.electrode_groups['probeA'].lfp_sampling_rate

    # Apply bandpass filter to each of the random channels separately
    lfp_df[channels] = lfp_df[channels].apply(
        lambda x: bandpass_filter_gamma(x, lowcut=35, highcut=65, fs=sampling_rate))

    # Calculate the median values down each column for the random channels
    median_values = lfp_df.loc[:, channels].median(axis=0)

    for i in range(len(presentation_df)):
        start_time = presentation_df.start_time[i] + interval[0]
        end_time = start_time + interval[1]

    # Calculate the start and end indices based on the sampling rate
    start_index = int(start_time * sampling_rate)
    end_index = int(end_time * sampling_rate)
    num_rows = end_index - start_index + 1

    # Create a new DataFrame to store the binary values within the desired time range
    binary_df = pd.DataFrame(index=range(num_rows), columns=channels)

    # Iterate over the columns (channels) of the random channels
    for column in channels:
        # Iterate over the rows (data points) within the desired time range
        for i, value in enumerate(lfp_df[column][start_index:end_index + 1]):
            # Check if the value is above or below the average for that channel
            if value > median_values[column]:
                binary_df.loc[i, column] = 1
            else:
                binary_df.loc[i, column] = 0

    allData = binary_df.to_numpy()
    allData = allData.astype(np.int16)

    # get all possible unique states from the data
    possible_states = np.unique(allData, axis=0)
    print(possible_states.shape)

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
    for iter_count in range(3):
        # Choose three random channels
        random_channels = random.sample(list(lfp_df.columns), 3)

        presentations = [flashes_presentations, static_gratings_presentations, gabors_presentations,
                         natural_scenes_presentations, natural_move_one_presentations,
                         natural_move_three_presentations]

        presentation_names = ['flashes', 'static_gratings', 'gabors', 'natural_scenes', 'natural_move_one', 'natural_move_three']

        # Store max phi and corresponding state for each presentation and interval
        presentation_phi_values = {}

        intervals = [(1, 50), (51, 100), (101, 150), (151, 200)]  # Define the intervals here

        for i, presentation in enumerate(presentations):
            # Store max phi and corresponding state for each interval
            interval_phi_values = {}

            for interval in intervals:
                max_phi, max_phi_state = compute_phi_lfp_rand_int(presentation, lfp_df, nwb, interval, random_channels)
                interval_phi_values[interval] = (max_phi, max_phi_state)
                print(f'Finished computing Phi value for {presentation_names[i]} - Interval {interval[0]}-{interval[1]} seconds')

            presentation_phi_values[presentation_names[i]] = interval_phi_values

        for presentation, intervals in presentation_phi_values.items():
            print(f'Presentation: {presentation}')
            for interval, phi_values in intervals.items():
                print(f'Interval: {interval[0]}-{interval[1]} seconds')
                print(f'Max Phi: {phi_values[0]}, Max Phi State: {phi_values[1]}')

        # Data Preparation
        presentations = list(presentation_phi_values.keys())  # Get presentation types
        intervals = sorted(set([i for sublist in list(presentation_phi_values.values()) for i in sublist.keys()]))  # Get all unique intervals

        data = {}
        for presentation in presentations:
            data[presentation] = [presentation_phi_values[presentation][interval][0] for interval in
                                intervals]  # Extracting Max Phi value for each interval

        # Create the graph
        fig, ax = plt.subplots()
        colors = ['red', 'green', 'blue', 'orange', 'purple',
              'yellow']  # This can be adjusted based on the number of presentations

        # Determine the spacing between each presentation type
        spacing_between_presentations = 0.4  # Adjust this value as desired
        spacing_within_presentation = 0.2  # Adjust this value as desired

        for i, presentation in enumerate(presentations):
            x_values = np.arange(len(intervals)) * spacing_within_presentation + i * (
                        len(intervals) * spacing_within_presentation + spacing_between_presentations)
            y_values = data[presentation]
            ax.scatter(x_values, y_values, label=presentation, color=colors[i])

        # Set x-axis labels and tick positions
        ax.set_xticks(np.arange(len(presentations)) * (len(intervals) * spacing_within_presentation + spacing_between_presentations) + (
                    len(intervals) - 1) * spacing_within_presentation / 2)
        ax.set_xticklabels(presentations, rotation=45)

        # Set axis labels and title
        ax.set_xlabel('Presentation Type (Duration 200s)')
        ax.set_ylabel('Phi-values')
        ax.set_title('Phi-values for Different Presentations')

        # Set legend with smaller font size
        legend = ax.legend(prop={'size': 8})

        # Save the figure with proper padding
        plt.savefig(f'phi_values_int_rand_gamma_{iter_count}.png', bbox_inches='tight')

        # Display the graph
        plt.show()

