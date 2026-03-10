# --- Reproducibility Config ---
RANDOM_SEED = 42

# --- Model Config ---
RWIN_SIZE = 20          # Time_step_TRES
DWIN_SIZE = 20          # Time_step_TCHA
INIT_SIZE = 100         # Initial dataset size
ALPHA = 0.01
MAXR = 10               # Maximum number of anomalies

# --- Training Config ---
EPOCHS = 1
EARLY_STOP = 3

# --- Apriori Config ---
APRIORI_SUPPORT = 0.05
VOTING_THRESHOLD = 1

# --- Parallel Processing ---
NUM_WORKERS = -1