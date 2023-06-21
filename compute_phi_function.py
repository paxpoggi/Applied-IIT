import pandas as pd
import numpy as np
import pyphi
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO

# Disable PyPhi progress bars and welcome message
pyphi.config.PROGRESS_BARS = False
pyphi.config.WELCOME_OFF = True

# Loading nwb_master file
filepath = '/Users/poggi/Documents/Maier Lab/NWB Data/sub-699733573_ses-715093703.nwb'
io = NWBHDF5IO(filepath, 'r',load_namespaces = True)  # open the file in read mode 'r'
nwb_master = io.read() # nwb dataset

# Opening ecephys probe file
filepath = '/Users/poggi/Documents/Maier Lab/NWB Data/sub-699733573_ses-715093703_probe-810755797_ecephys.nwb'
io = NWBHDF5IO(filepath, 'r',load_namespaces = True)  # open the file in read mode 'r'
nwb = io.read() # nwb dataset

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

def compute_phi_lfp_og(presentation_df, lfp_df, nwb):
    for i in range(len(presentation_df)):
        one_second = presentation_df.start_time[i] + 1
        fifty_seconds = presentation_df.start_time[i] + 50
        one_hundred_seconds = presentation_df.start_time[i] + 100
        one_hundred_fifty_seconds = presentation_df.start_time[i] + 150
        two_hundred_seconds = presentation_df.start_time[i] + 200
        # Calculate the variation for each channel (column) using the std() function
    channel_variations = lfp_df.std(axis=0)

    # Sort the channels based on their variation values in descending order
    sorted_channels = channel_variations.sort_values(ascending=False)

    # Get the top three channels with the most variation
    top_three_channels = sorted_channels.head(3)

    # Calculate the average values down each column for the top three channels
    median_values = lfp_df.loc[:, top_three_channels.index].median(axis=0)

    # Retrieve the sampling rate from the NWB file
    sampling_rate = nwb.electrode_groups['probeA'].lfp_sampling_rate

    # Convert time to index based on the sampling rate
    start_index = int(one_second * sampling_rate)
    end_index = int(fifty_seconds * sampling_rate)
    num_rows = end_index - start_index + 1

    # Create a new DataFrame to store the binary values within the desired time range
    binary_df = pd.DataFrame(index=range(num_rows), columns=top_three_channels.index)

    # Iterate over the columns (channels) of the top three channels
    for column in top_three_channels.index:
        # Iterate over the rows (data points) within the desired time range
        for i, value in enumerate(lfp_df[column][start_index:end_index + 1]):
            # Check if the value is above or below the average for that channel
            if value > median_values[column]:
                binary_df.loc[i, column] = 1
            else:
                binary_df.loc[i, column] = 0

    allData = binary_df.to_numpy()
    allData = allData.astype(np.int64)

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
    presentations = [flashes_presentations, static_gratings_presentations, gabors_presentations,compute
                     natural_scenes_presentations, natural_move_one_presentations,
                     natural_move_three_presentations]

    presentation_names = ['flashes', 'static_gratings', 'gabors', 'natural_scenes', 'natural_move_one', 'natural_move_three']

    # Store max phi and corresponding state for each presentation
    presentation_phi_values = {}

    for i, presentation in enumerate(presentations):
        average_phi, max_phi_state = compute_phi_lfp_og(presentation, lfp_df, nwb)
        presentation_phi_values[i] = (average_phi, max_phi_state)
        print(f'Finished computing Phi value for {presentation_names[i]}')

    for key, value in presentation_phi_values.items():
        print(f'Presentation {key}: Max Phi = {value[0]}, Max Phi State = {value[1]}')
