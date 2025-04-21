import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import pandas as pd

# --------------------------
# Streamlit UI Controls
# --------------------------
st.title("📱 Dynamic Frequency Hopping vs Smart Jammers (Advanced)")

# Sliders for simulation parameters
num_channels = st.sidebar.slider("Frequency Channels", 3, 10, 5)
num_users = st.sidebar.slider("Users", 1, 3, 2)
num_jammers = st.sidebar.slider("Adaptive DL Jammers", 1, 4, 2)
num_rounds = st.sidebar.slider("Simulation Rounds", 500, 3000, 1000, step=500)

# Hyperparameters
alpha = st.sidebar.slider("Learning Rate (α)", 0.001, 0.1, 0.01, step=0.001)
gamma = st.sidebar.slider("Discount Factor (γ)", 0.5, 0.99, 0.9, step=0.01)
epsilon = st.sidebar.slider("Exploration Rate (ε)", 0.0, 1.0, 0.1, step=0.05)
memory_window = st.sidebar.slider("Adaptive Jammer Memory Window", 10, 300, 100, step=10)

# Constants
batch_size = 32
buffer_size = 1000
update_every = 10

# --------------------------
# DL Model
# --------------------------
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dl_models = [DQN(num_channels * 2, num_channels).to(device) for _ in range(num_jammers)]
optimizers = [optim.Adam(model.parameters(), lr=alpha) for model in dl_models]
replay_buffers = [deque(maxlen=buffer_size) for _ in range(num_jammers)]
loss_fn = nn.MSELoss()

# --------------------------
# Q-Learning Tables
# --------------------------
q_tables_user = [np.zeros(num_channels) for _ in range(num_users)]
q_tables_ml = [np.zeros(num_channels) for _ in range(num_users)]

# --------------------------
# Simulation Buffers
# --------------------------
user_history = [[] for _ in range(num_users)]
jammer_history = [[] for _ in range(num_jammers)]
success_rates_ml = [[] for _ in range(num_users)]
success_rates_dl = [[[] for _ in range(num_jammers)] for _ in range(num_users)]
success_rates_users_rl = [[] for _ in range(num_users)]  # RL strategy success rates
results_log = []

# --------------------------
# Simulation
# --------------------------
for t in range(num_rounds):
    # User choices
    user_choices = []
    for u in range(num_users):
        if np.random.rand() < epsilon:
            choice = np.random.randint(num_channels)  # Exploration
        else:
            choice = np.argmax(q_tables_user[u])  # Exploitation
        user_choices.append(choice)
        user_history[u].append(choice)  # Log history

    # --------------------------
    # ML Jammers
    # --------------------------
    for u in range(num_users):
        choice_ml = np.argmax(q_tables_ml[u]) if np.random.rand() > epsilon else np.random.randint(num_channels)
        success_ml = int(user_choices[u] != choice_ml)
        reward_ml = 1 - success_ml
        q_tables_ml[u][choice_ml] += alpha * (reward_ml + gamma * np.max(q_tables_ml[u]) - q_tables_ml[u][choice_ml])
        success_rates_ml[u].append(success_ml)

        results_log.append({
            "Round": t, "User": f"User_{u+1}", "Freq_Used": user_choices[u],
            "Jammer_Type": "ML", "Jammer_ID": "ML", "Freq_Jammed": choice_ml, "Success": success_ml
        })

    # --------------------------
    # DL Jammers
    # --------------------------
    for j in range(num_jammers):
        for u in range(num_users):
            # Use limited history for frequency distribution
            history = user_history[u][-min(len(user_history[u]), memory_window):]
            freq_dist = np.bincount(history, minlength=num_channels) / max(len(history), 1)
            current_freq = np.zeros(num_channels)
            current_freq[user_choices[u]] = 1
            state = torch.tensor(np.concatenate([current_freq, freq_dist]), dtype=torch.float32).to(device)

            # Select action
            if np.random.rand() < epsilon:
                action = np.random.randint(num_channels)
            else:
                with torch.no_grad():
                    action = torch.argmax(dl_models[j](state)).item()

            # Update jammer history and success rates
            jammer_history[j].append(action)
            success = int(user_choices[u] != action)
            reward = 1 - success
            success_rates_dl[u][j].append(success)
            replay_buffers[j].append((state.cpu(), action, reward))

            results_log.append({
                "Round": t, "User": f"User_{u+1}", "Freq_Used": user_choices[u],
                "Jammer_Type": "DL", "Jammer_ID": f"DL_{j+1}", "Freq_Jammed": action, "Success": success
            })

            # Train DL Model
            if len(replay_buffers[j]) >= batch_size and t % update_every == 0:
                batch = random.sample(replay_buffers[j], batch_size)
                states, actions, rewards = zip(*batch)
                states = torch.stack(states).to(device)
                actions = torch.tensor(actions).long().to(device)
                rewards = torch.tensor(rewards).float().to(device)

                q_values = dl_models[j](states)
                target_q = q_values.clone().detach()
                for idx, a in enumerate(actions):
                    target_q[idx, a] = rewards[idx] + gamma * torch.max(q_values[idx]).detach()

                output_q = dl_models[j](states)
                loss = loss_fn(output_q, target_q)
                optimizers[j].zero_grad()
                loss.backward()
                optimizers[j].step()

    # --------------------------
    # Update Q-Tables for Users
    # --------------------------
    for u in range(num_users):
        freq = user_choices[u]
        was_jammed = any(
            (log["Round"] == t and log["User"] == f"User_{u+1}" and log["Freq_Used"] == freq and log["Success"] == 0)
            for log in results_log if log["Round"] == t
        )
        reward = 1 if not was_jammed else 0
        q_tables_user[u][freq] += alpha * (reward + gamma * np.max(q_tables_user[u]) - q_tables_user[u][freq])
        success_rates_users_rl[u].append(reward)

# --------------------------
# Visualizations
# --------------------------
# (Remaining visualizations are unchanged; refer to your original implementation)
