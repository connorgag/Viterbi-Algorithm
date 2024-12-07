import math
import numpy as np
import matplotlib.pyplot as plt

def plot_data(title, x_label, y_label, x, y, alphabet):
    plt.scatter(x, y, marker='o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.yticks(range(len(alphabet)), list(alphabet))

    plt.savefig(title + ".png")

    plt.show()

def read_observations():
    with open('observations.txt', 'r') as file:
        return [int(i) for i in file.read().split()]


def read_transition_matrix():
    t_matrix = []
    with open('transitionMatrix.txt', 'r') as file:
        for line in file:
            row = [float(i) for i in line.split()]
            t_matrix.append(row)
    return t_matrix

def read_initial_state_distribution():
    with open('initialStateDistribution.txt', 'r') as file:
        return [float(line) for line in file]

def read_emission_matrix():
    e_matrix = []
    with open('emissionMatrix.txt', 'r') as file:
        for line in file:
            row = [float(i) for i in line.split()]
            e_matrix.append(row)
    return e_matrix


def initialize_matrix(initial_state_distributions, e_matrix, observations):
    result = [[None for i in range(len(observations))] for j in range(len(initial_state_distributions))]

    for i in range(len(initial_state_distributions)):
        prob_of_start_hidden_state = initial_state_distributions[i] # prob of starting at this hidden state
        emission_prob = e_matrix[i][observations[0]] # prob of going from hidden start state of i to current observation value 0

        result[i][0] = math.log(prob_of_start_hidden_state) + math.log(emission_prob)

    return result


def viterbi():
    observations = read_observations() # 1 x 430,000
    transition_matrix = read_transition_matrix() # 27 x 27
    initial_state_distributions = read_initial_state_distribution() # 0 to 26
    # position [12][1] shows the probability of going from hidden state 12 to observed value 1
    e_matrix = read_emission_matrix() # 2 x 27

    # Resulting shape of the matrix should be n x t
    # where n is the number of options for hidden states and t is the number of hidden/observed nodes
    alpha_matrix = initialize_matrix(initial_state_distributions, e_matrix, observations)
    print(len(alpha_matrix[0]))

    for t in range(1, len(observations)):
        # Calculates for row_index, t
        for row_index in range(len(alpha_matrix)):
            # Loop to find the max
            max_row_val = float('-inf')
            for inner_loop_row_index in range(len(alpha_matrix)):
                previous_state_prob = alpha_matrix[inner_loop_row_index][t-1]
                transition_prob = transition_matrix[inner_loop_row_index][row_index]
                previous_state_to_current_state_prob = previous_state_prob + math.log(transition_prob)
                
                if (previous_state_to_current_state_prob > max_row_val):
                    max_row_val = previous_state_to_current_state_prob
            
            emission_prob = math.log(e_matrix[row_index][observations[t]])

            alpha_matrix[row_index][t] = max_row_val + emission_prob

        if (t % 50000 == 0 or t == 429999):
            print("Iteration: " + str(t))
        
    return alpha_matrix


def backtrack(viterbi_matrix):
    transition_matrix = read_transition_matrix() # 27 x 27

    hidden_states = []
    num_hidden_states = len(viterbi_matrix)
    num_observations = len(viterbi_matrix[0])

    last_column = [viterbi_matrix[i][-1] for i in range(num_hidden_states)]
    last_hidden_state = np.argmax(last_column)
    hidden_states.insert(0, last_hidden_state)

    for t in range(num_observations - 1, -1, -1):
        alpha_column = [viterbi_matrix[one_hidden_state][t] + math.log(transition_matrix[one_hidden_state][last_hidden_state]) for one_hidden_state in range(num_hidden_states)]
        last_hidden_state = np.argmax(alpha_column)
        hidden_states.insert(0, last_hidden_state)
        
    return hidden_states


def get_hidden_phrase(hidden_states):
    hidden_phrase = ""
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    for i in range(len(hidden_states)):
        if (i == 0 or alphabet[hidden_states[i]] != hidden_phrase[-1]):
            hidden_phrase = hidden_phrase + alphabet[hidden_states[i]]

    return hidden_phrase


def plot_hidden_states_at_different_times(alpha_matrix):
    most_likely_states = []
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    
    # Go through all of the observations
    for t in range(len(alpha_matrix[0])):
        most_likely_states.append(np.argmax([i[t] for i in alpha_matrix]))
        
    x_axis = range(len(most_likely_states))
    
    plot_data("Most Likely Unobserved State at Time t", "Observation t", "Most Likely State", x_axis, most_likely_states, alphabet)



def main():
    # Get the hidden phrase
    alpha_matrix = viterbi()
    hidden_states = backtrack(alpha_matrix)
    hidden_phrase = get_hidden_phrase(hidden_states)
    print(hidden_phrase)

    plot_hidden_states_at_different_times(alpha_matrix)



main()
