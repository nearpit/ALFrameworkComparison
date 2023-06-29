RANDOM_STATE = 42
DUMMY_VAR = 1e-5
DUMMY_ARR = [10, 5]

# Active Learning
BUDGET = 1000
HINDERED_ITERATIONS = 10 

# Monte-Carlo BALD
DROPOUT_RATE = 0.2

# MLP HPO 
DECAY_RANGE = [1e-7, 1e-1]
DEPTH_RANGE = [1, 5]
WIDHT_RANGE = [4, 92]
LR_RANGE = [1e-4, 1e-1]

# General DL
OPTIMIZER = "Adam"
EPOCHS = 500
PATIENCE = 50
MIN_DELTA = 0

#MLP Specifics
MLP_LAST_ACTIVATION = "Softmax"
MLP_CRITERION = "CrossEntropyLoss"

# AE Spesifics
AE_LAST_ACTIVATION = "Identity"
AE_CRITERION = "MSELoss"

# Optuna HPO
N_STARTUP_TRIALS = 10
N_WARMUP_STEPS = 100
TOTAL_TRIALS = 100
OPTUNA_PRUNER = {"name": "MedianPruner",
          "configs": {
              "n_startup_trials": N_STARTUP_TRIALS,
              "n_warmup_steps": N_WARMUP_STEPS
          }}
OPTUNA_SAMPLER = {"name": "TPESampler",
                  "configs": {}}   
# DEBUG PARAMS
# EPOCHS = 500
TOTAL_TRIALS = 2