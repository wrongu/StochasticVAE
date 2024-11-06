import os

##################
## Model config ##
##################

PLAN = [500, 300, 200, 100, 50]
PLAN_DECODER = [50, 100, 300, 500]
LATENT_DIM = 20

#####################
## Training config ##
#####################

LEARNING_RATE = 1e-5
EPOCHS = 100
BATCH_SIZE = 128

########################
## Environment config ##
########################

# Only let lightning 'see' one GPU, but can be overridden by setting the environment variable
# CUDA_VISIBLE_DEVICES from outside the script.
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")
MLFLOW_TRACKING_URI = "/data/projects/SVAE/mlruns/"
DATA_ROOT = "/data/datasets/"
