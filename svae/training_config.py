import torch

PLAN = [500, 300, 200, 100, 50]
PLAN_DECODER = [50, 100, 300, 500]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1e-4
EPOCHS = 10
BATCH_SIZE = 128
LATENT_DIM = 20