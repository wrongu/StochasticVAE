import os
from argparse import ArgumentParser
from pathlib import Path

import lightning as lit
import mlflow
import torch
import torchvision.datasets as datasets
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from stochastic_density_network import Stochastic_Density_NN
from stochastic_recognition_model import Stochastic_Recognition_NN
from stochastic_vae import Stochastic_VAE


torch.set_float32_matmul_precision("high")
ENCODER_PLAN = [500, 300, 200, 100, 50]
DECODER_PLAN = [50, 100, 300, 500]
MLFLOW_TRACKING_URI = "file:///data/projects/SVAE/mlruns"
MLFLOW_EXPERIMENT = "LitSVAE_RDL"
DATA_ROOT = "/data/datasets/"


def main(
    latent_dim: int = 20,
    lambda_: float = 2.0,
    number_of_nearest_neighbors: int = 4,
    n_forward_pass: int = 8,
    learning_rate: float = 1e-5,
    epochs: int = 200,
    batch_size: int = 250,
    user_input_logvar: float = -10,
    ablate_entropy: bool = False,
    ablate_fim: bool = False,
    load_model_from_run: str = None,
    init_encoder: bool = False,
    test_on_synthetic_data: bool = False,
):
    ################
    ## Data setup ##
    ################

    train_dataset = datasets.MNIST(
        root=Path(DATA_ROOT) / "mnist", train=True, transform=transforms.ToTensor()
    )
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [50000, 10000], generator=torch.Generator().manual_seed(123456)
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    #################
    ## Model setup ##
    #################

    svae = Stochastic_VAE(
        Stochastic_Recognition_NN(
            input_dim=784,
            latent_dim=latent_dim,
            user_input_logvar=user_input_logvar,
            plan=ENCODER_PLAN,
        ),
        Stochastic_Density_NN(
            input_dim=784,
            latent_dim=latent_dim,
            plan=DECODER_PLAN,
        ),
        lambda_=lambda_,
        lr=learning_rate,
        k_neighbor=number_of_nearest_neighbors,
        n_forward=n_forward_pass,
        ablate_entropy=ablate_entropy,
        ablate_fim=ablate_fim,
    )
    #####################
    ## Lightning setup ##
    #####################

    logger = MLFlowLogger(
        experiment_name=MLFLOW_EXPERIMENT,
        tracking_uri=MLFLOW_TRACKING_URI,
        log_model=True,
    )
    trainer = lit.Trainer(
        logger=logger,
        max_epochs=epochs,
        default_root_dir=logger.root_dir,  # TODO - double check that logger.root_dir is right
        callbacks=ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
    )

    ############
    ## Run it ##
    ############

    logger.log_hyperparams(
        {
            "encoder_plan": ENCODER_PLAN,
            "decoder_plan": DECODER_PLAN,
            "latent_dim": latent_dim,
            "lambda_": lambda_,
            "number_of_nearest_neighbors": number_of_nearest_neighbors,
            "n_forward_pass": n_forward_pass,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "user_input_logvar": user_input_logvar,
            "ablate_entropy": ablate_entropy,
            "ablate_fim": ablate_fim,
            "decoder_source": load_model_from_run,
            "init_encoder": init_encoder,
            "test_on_synthetic_data": test_on_synthetic_data,
        }
    )

    # If specified, load decoder weights from a checkpoint and freeze it
    if load_model_from_run:
        checkpoint = torch.load(get_best_checkpoint_for_run(load_model_from_run))

        def _keep_param(param_name):
            return param_name.startswith("decoder") or (init_encoder and "logvar" not in param_name)

        state_dict_to_load = {k: v for k, v in checkpoint["state_dict"].items() if _keep_param(k)}

        svae.load_state_dict(state_dict_to_load, strict=False)
        for param in svae.decoder.parameters():
            param.requires_grad = False

    sanity_check_params = {k: v.detach().clone() for k, v in svae.named_parameters()}

    # Do training
    trainer.fit(model=svae, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if load_model_from_run:
        # Assert that the decoder weights are unchanged and the encoder params are changed
        for name, param in svae.named_parameters():
            if name.startswith("decoder"):
                assert torch.equal(
                    param, sanity_check_params[name]
                ), f"Decoder param {name} changed"
            elif name.startswith("encoder"):
                assert (not torch.equal(param, sanity_check_params[name])) or (
                    not torch.any(torch.isfinite(param))
                ), f"Encoder param {name} unchanged"
            else:
                raise ValueError(f"Unknown parameter name: {name}")

    # (Maybe) generate a synthetic dataset using the model's decoder
    if test_on_synthetic_data:
        gen_x = svae.decoder.generate(n=10000, pixel_noise=True)
        gen_labels = torch.randint(0, 10, (10000,))
        test_dataset = TensorDataset(gen_x, gen_labels)
    else:
        test_dataset = datasets.MNIST(
            root=Path(DATA_ROOT) / "mnist", train=False, transform=transforms.ToTensor()
        )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Before testing, re-load the best checkpoint from training
    best_checkpoint = torch.load(get_best_checkpoint_for_run(logger.run_id))
    svae.load_state_dict(best_checkpoint["state_dict"])

    # Do testing (including inference-goodness)
    trainer.test(model=svae, dataloaders=test_loader)


def get_best_checkpoint_for_run(run_id: str):
    checkpoints_dir = Path(mlflow.artifacts.download_artifacts(run_id=run_id))
    for meta_file in checkpoints_dir.glob("**/aliases.txt"):
        with open(meta_file, "r") as f:
            aliases = f.read()
        if "best" in aliases:
            checkpoint_file = next(meta_file.parent.glob("*.ckpt"))
            break
    else:
        raise FileNotFoundError("No best checkpoint file found in the specified run.")
    return checkpoint_file


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=20),
    parser.add_argument("--lambda", dest="lambda_", type=float, default=2.0),
    parser.add_argument("--number_of_nearest_neighbors", type=int, default=4),
    parser.add_argument("--n_forward_pass", type=int, default=8),
    parser.add_argument("--learning_rate", type=float, default=1e-3),
    parser.add_argument("--epochs", type=int, default=100),
    parser.add_argument("--batch_size", type=int, default=250),
    parser.add_argument("--user_input_logvar", type=float, default=-10),
    parser.add_argument("--ablate_entropy", type=bool, default=False),
    parser.add_argument("--ablate_fim", type=bool, default=False),
    parser.add_argument("--load_model_from_run", type=str, default=None),
    parser.add_argument("--init_encoder", action="store_true", default=False)
    parser.add_argument("--test_on_synthetic_data", action="store_true", default=False)
    args = parser.parse_args()

    # Only let lightning 'see' one GPU, but can be overridden by setting the environment variable
    # CUDA_VISIBLE_DEVICES from outside the script.
    os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")

    main(**vars(args))
