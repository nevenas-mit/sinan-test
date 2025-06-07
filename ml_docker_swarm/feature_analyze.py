import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Setup Paths ===
data_dir = '/home/jovans2/test_llms/sinan-test/docker_swarm/logs/collected_data/dataset'
PLOT_DIR = '/home/jovans2/test_llms/sinan-test/ml_docker_swarm/distribution_figs'
os.makedirs(PLOT_DIR, exist_ok=True)

# === Load Data ===
sys_data = np.load(os.path.join(data_dir, 'sys_data_train.npy'))  # (N, 6, 28, 5)
lat_data = np.load(os.path.join(data_dir, 'lat_data_train.npy'))  # (N, 5, 5)
nxt_data = np.squeeze(np.load(os.path.join(data_dir, 'nxt_k_data_train.npy')))[:, :, 0]  # (N, 28)
label_data = np.squeeze(np.load(os.path.join(data_dir, 'nxt_k_train_label.npy')))[:, :, 0]  # (N, 5)

print(f"System Data shape: {sys_data.shape}")

# === Feature Names ===
metric_names = ["RPS", "REPLICA", "CPU_LIM", "CPU_USE", "RSS", "CACHE"]
service_names = [
    'compose-post-redis', 'compose-post-service', 'home-timeline-redis',
    'home-timeline-service', 'nginx-thrift', 'post-storage-memcached',
    'post-storage-mongodb', 'post-storage-service', 'social-graph-mongodb',
    'social-graph-service', 'text-service', 'unique-id-service', 'url-shorten-service',
    'user-memcached', 'user-mongodb', 'user-service', 'media-service', 'media-frontend',
    'user-timeline-service', 'user-timeline-redis', 'follow-service', 'follow-redis',
    'write-home-timeline-rabbitmq', 'write-home-timeline-redis',
    'write-home-timeline-service', 'read-home-timeline-redis',
    'read-home-timeline-service', 'jaeger'
]

# === Plot Distributions ===
# sys_data: (N, 6 metrics, 28 services, 5 time steps)
N, M, S, T = sys_data.shape

for m in range(M):
    for s in range(S):
        values = sys_data[:, m, s, :].reshape(-1)  # Flatten over N and T

        plt.figure(figsize=(8, 4))
        sns.histplot(values, bins=100, kde=True)
        plt.title(f"Distribution of {metric_names[m]} for service '{service_names[s]}'")
        plt.xlabel(f"{metric_names[m]} value")
        plt.ylabel("Frequency")

        fname = f"metric_{metric_names[m].replace('%','pct')}_service_{service_names[s].replace('-', '_')}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, fname))
        plt.close()

print("Feature distribution plots saved to:", PLOT_DIR)

# === Latency Distribution Plots ===
# lat_data: (N, 5 latency metrics, 5 time steps)
latency_metric_names = ['P90', 'P95', 'P98', 'P99', 'P999']

N_lat, M_lat, T_lat = lat_data.shape
assert N == N_lat, "Mismatch in sample count between sys_data and lat_data"

for m in range(M_lat):
    values = lat_data[:, m, :].reshape(-1)  # Flatten over N and T

    plt.figure(figsize=(8, 4))
    sns.histplot(values, bins=100, kde=True)
    plt.title(f"Distribution of Latency Metric: {latency_metric_names[m]}")
    plt.xlabel("Latency value (ms or appropriate unit)")
    plt.ylabel("Frequency")

    fname = f"latency_metric_{m+1}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, fname))
    plt.close()

print("Latency distribution plots saved to:", PLOT_DIR)