import streamlit as st
import numpy as np
import random

# Define Smart Grid Environment
class SmartGridEnv:
    def __init__(self, num_states=5, num_actions=3):
        self.num_states = num_states
        self.num_actions = num_actions
        self.state = random.randint(0, num_states - 1)

    def step(self, action):
        if action == 0:  # Reduce load
            reward = -abs(self.state - 1)
        elif action == 1:  # Maintain load
            reward = -abs(self.state - 2)
        else:  # Increase load
            reward = -abs(self.state - 3)

        self.state = min(max(self.state + (action - 1), 0), self.num_states - 1)
        return self.state, reward

    def reset(self):
        self.state = random.randint(0, self.num_states - 1)
        return self.state


# Load trained Q-table (assuming it is saved)
num_states = 5
num_actions = 3
Q_table = np.load("q_table.npy")  # Load trained Q-table

# Initialize environment
env = SmartGridEnv(num_states, num_actions)

# Streamlit UI
st.title("⚡ Smart Grid Load Balancing - RL Model")

# Display Environment State
state = env.reset()
st.write(f"**Initial Load Level (State):** {state}")

# User Input for Action
action = st.radio("Select Action", ["Reduce Load (0)", "Maintain Load (1)", "Increase Load (2)"])
action = int(action.split()[2][1])  # Extract number

# Take action and show result
if st.button("Run Simulation"):
    next_state, reward = env.step(action)
    st.write(f"**New Load Level (State):** {next_state}")
    st.write(f"**Reward Received:** {reward}")

    # Show Q-table values for this state
    st.write("**Q-Table Values for Current State:**")
    st.write(Q_table[state])

# Footer
st.sidebar.write("Developed using Reinforcement Learning & Streamlit 🚀")
