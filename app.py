import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import seaborn as sns
import pandas as pd

# --------------------------
# Streamlit UI Controls
# --------------------------
st.title("📱 Dynamic Frequency Hopping vs Smart Jammers (Advanced)")

num_channels = st.sidebar.slider("Frequency Channels", 3, 10, 5)
num_users = st.sidebar.slider("Users", 1, 3, 2)
num_jammers = st.sidebar.slider("Adaptive DL Jammers", 1, 4, 2)
num_rounds = st.sidebar.slider("Simulation Rounds", 500, 3000, 1000, step=500)

# Hyperparameters with sliders
alpha = st.sidebar.slider("Learning Rate (α)", 0.001, 0.1, 0.01, step=0.001)
gamma = st.sidebar.slider("Discount Factor (γ)", 0.5, 0.99, 0.9, step=0.01)
epsilon = st.sidebar.slider("Exploration Rate (ε)", 0.0, 1.0, 0.1, step=0.05)
memory_window = st.sidebar.slider("Adaptive Jammer Memory Window", 10, 300, 100, step=10)

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
# Simulation Buffers
# --------------------------
user_strategy = np.ones(num_channels) / num_channels
q_tables_ml = [np.zeros(num_channels) for _ in range(num_users)]
success_rates_ml = [[] for _ in range(num_users)]
success_rates_dl = [[[] for _ in range(num_jammers)] for _ in range(num_users)]

user_history = [[] for _ in range(num_users)]
jammer_history = [[] for _ in range(num_jammers)]

results_log = []

# --------------------------
# Simulation
# --------------------------
for t in range(num_rounds):
    user_choices = [np.random.choice(num_channels, p=user_strategy) for _ in range(num_users)]
    for u in range(num_users):
        user_history[u].append(user_choices[u])

    # ML Jammers
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
    
    

    # DL Jammers
    for j in range(num_jammers):
        for u in range(num_users):
            history = user_history[u][-memory_window:]
            freq_dist = np.bincount(history, minlength=num_channels) / memory_window
            current_freq = np.zeros(num_channels)
            current_freq[user_choices[u]] = 1
            state = torch.tensor(np.concatenate([current_freq, freq_dist]), dtype=torch.float32).to(device)

            if np.random.rand() < epsilon:
                action = np.random.randint(num_channels)
            else:
                with torch.no_grad():
                    action = torch.argmax(dl_models[j](state)).item()

            jammer_history[j].append(action)
            success = int(user_choices[u] != action)
            reward = 1 - success
            success_rates_dl[u][j].append(success)
            replay_buffers[j].append((state.cpu(), action, reward))

            results_log.append({
                "Round": t, "User": f"User_{u+1}", "Freq_Used": user_choices[u],
                "Jammer_Type": "DL", "Jammer_ID": f"DL_{j+1}", "Freq_Jammed": action, "Success": success
            })

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
# Visualizations
# --------------------------
def rolling_avg(data, window=50):
    return np.convolve(data, np.ones(window) / window, mode='valid')

st.subheader("📈 Success Rate Over Time")
fig, ax = plt.subplots(figsize=(12, 6))
for u in range(num_users):
    ax.plot(rolling_avg(success_rates_ml[u]), label=f'User {u+1} vs ML Jammer')
    for j in range(num_jammers):
        ax.plot(rolling_avg(success_rates_dl[u][j]), linestyle='--', label=f'User {u+1} vs DL Jammer {j+1}')
ax.set_xlabel("Time Slot")
ax.set_ylabel("Success Rate")
ax.set_title("Transmission Success Rates")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Frequency Conflict Distribution (Bar Chart)
st.subheader("📊 Frequency Conflict Distribution")
jammed_counts = np.zeros(num_channels)
clean_counts = np.zeros(num_channels)

for row in results_log:
    freq = row["Freq_Used"]
    if row["Success"] == 0:
        jammed_counts[freq] += 1
    else:
        clean_counts[freq] += 1

freq_labels = [f"F{f}" for f in range(num_channels)]
fig, ax = plt.subplots()
ax.bar(freq_labels, jammed_counts, color='red', label="Jammed")
ax.bar(freq_labels, clean_counts, bottom=jammed_counts, color='green', label="Not Jammed")
ax.set_ylabel("Counts")
ax.set_title("Frequency Conflict: Jammed vs Not Jammed")
ax.legend()
st.pyplot(fig)

# Pie Chart for Frequency Usage Distribution (for users)
st.subheader("🍰 Frequency Usage Distribution (Users)")
user_freq_usage = np.zeros(num_channels)
for u in range(num_users):
    user_freq_usage += np.bincount(user_history[u], minlength=num_channels)

fig, ax = plt.subplots()
ax.pie(user_freq_usage, labels=[f"F{f}" for f in range(num_channels)], autopct='%1.1f%%', startangle=90)
ax.set_title("User Frequency Usage Distribution")
st.pyplot(fig)
st.subheader("📊 Stacked Success Rate Trend (Smoothed)")

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(rolling_avg(success_rates_ml[0])))

for u in range(num_users):
    stacked_data = [rolling_avg(success_rates_ml[u])]
    for j in range(num_jammers):
        stacked_data.append(rolling_avg(success_rates_dl[u][j]))
    stacked_data = np.vstack(stacked_data)
    ax.stackplot(x, stacked_data, labels=[f'User {u+1} - ML'] + [f'User {u+1} - DL {j+1}' for j in range(num_jammers)])

ax.set_title("Stacked Success Rate Trend Over Time")
ax.set_xlabel("Time Slot")
ax.set_ylabel("Smoothed Success Rate")
ax.legend(loc='upper right')
st.pyplot(fig)

st.subheader("🧠 DL Jammers Performance Comparison")

dl_success_data = []
for j in range(num_jammers):
    for u in range(num_users):
        dl_success_data.append({
            "Jammer": f"DL_{j+1}",
            "User": f"User_{u+1}",
            "SuccessRate": np.mean(success_rates_dl[u][j])
        })

df_dl_success = pd.DataFrame(dl_success_data)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x="Jammer", y="SuccessRate", hue="User", data=df_dl_success, ax=ax)
ax.set_title("DL Jammer Success Rate Comparison Across Users")
ax.set_ylabel("Avg. Success Rate")
st.pyplot(fig)

# Cumulative Success Rate Curve
# st.subheader("📊 Cumulative Success Rate")
# cumulative_success_ml = np.cumsum([np.sum(success_rates_ml[u]) for u in range(num_users)])
# cumulative_success_dl = np.cumsum([[np.sum(success_rates_dl[u][j]) for j in range(num_jammers)] for u in range(num_users)])

# fig, ax = plt.subplots(figsize=(12, 6))
# for u in range(num_users):
#     ax.plot(cumulative_success_ml[u], label=f'User {u+1} Cumulative Success (ML)', linestyle='-', marker='o')
#     for j in range(num_jammers):
#         ax.plot(cumulative_success_dl[u][j], label=f'User {u+1} Cumulative Success (DL {j+1})', linestyle='--', marker='x')
# ax.set_xlabel("Time Slot")
# ax.set_ylabel("Cumulative Success")
# ax.set_title("Cumulative Success Rate Over Time")
# ax.legend()
# ax.grid(True)
# st.pyplot(fig)
# Correct initialization for cumulative success rates
cumulative_success_dl = [[0] * num_jammers for _ in range(num_users)]

# During the simulation, we need to accumulate the success rates for each user and jammer:
for u in range(num_users):
    for j in range(num_jammers):
        cumulative_success_dl[u][j] += np.sum(success_rates_dl[u][j])

# Correct plotting for the cumulative success rate
st.subheader("📊 Cumulative Success Rate")
fig, ax = plt.subplots(figsize=(12, 6))

# Plot for ML jammer success
for u in range(num_users):
    ax.plot(np.cumsum(success_rates_ml[u]), label=f'User {u+1} Cumulative Success (ML)', linestyle='-', marker='o')

# Plot for DL jammer success
for u in range(num_users):
    for j in range(num_jammers):
        ax.plot(np.cumsum(success_rates_dl[u][j]), label=f'User {u+1} Cumulative Success (DL {j+1})', linestyle='--', marker='x')

ax.set_xlabel("Time Slot")
ax.set_ylabel("Cumulative Success")
ax.set_title("Cumulative Success Rate Over Time")
ax.legend()
ax.grid(True)
st.pyplot(fig)


# CSV Export
df_log = pd.DataFrame(results_log)
st.subheader("📄 Simulation Logs")
st.dataframe(df_log.tail(10))
csv = df_log.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Full Log as CSV", csv, "frequency_jamming_log.csv", "text/csv")

# Auto-report Summary
st.subheader("📆 Auto-report Summary")
for u in range(num_users):
    total_success = np.sum(success_rates_ml[u])
    st.write(f"User {u+1} ML Jammer Success Rate: {100 * total_success / num_rounds:.2f}%")
    for j in range(num_jammers):
        total_success_dl = np.sum(success_rates_dl[u][j])
        st.write(f"User {u+1} vs DL Jammer {j+1} Success Rate: {100 * total_success_dl / num_rounds:.2f}%")

# End of code
