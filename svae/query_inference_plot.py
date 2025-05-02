from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow.tracking import MlflowClient
from nn_lib.utils import search_runs_by_params
from torchvision import datasets, transforms

from main import DATA_ROOT
from stochastic_density_network import PixelCovariance

client = MlflowClient()


def barplot_with_custom_errors(data, x, y, yerr, **kwargs):
    data_low = data.copy()
    data_hi = data.copy()
    data_low[y] = data[y] - data[yerr]
    data_hi[y] = data[y] + data[yerr]
    data_combo = pd.concat([data_low, data, data_hi], axis=0).reset_index()

    def calculate_errors(low_mid_hi):
        assert len(low_mid_hi) == 3
        return np.min(low_mid_hi), np.max(low_mid_hi)

    return sns.barplot(data_combo, x=x, y=y, errorbar=calculate_errors, **kwargs)


def plot_inference_goodness(plot_df):
    ds = datasets.MNIST(
        root=Path(DATA_ROOT) / "mnist", train=False, transform=transforms.ToTensor()
    )

    plot_df["goodness_standard_error"] = np.sqrt(
        (plot_df["metrics.goodness_moment2"] - (plot_df["metrics.goodness_moment1"] ** 2)) / len(ds)
    )
    plot_df["goodness_relative"] = (
        plot_df["metrics.goodness_moment1"]
        - plot_df[np.isinf(plot_df["lambda_f"])]["metrics.goodness_moment1"].values[0]
    )

    plt.figure(figsize=(8, 6))
    barplot_with_custom_errors(
        data=plot_df,
        x="params.lambda_",
        y="goodness_relative",
        yerr="goodness_standard_error",
        hue="params.user_input_logvar",
    )
    plt.xlabel("Lambda")
    plt.ylabel("$KL(m(z|x)||p(z|x)) + C$")
    plt.title("Inference Error")
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig("plots/lambda_v_inference_goodness_mcse.png", dpi=300)


def plot_metric(plot_df, metric, human_readable_name, ylim=None):
    sns.barplot(plot_df, x="params.lambda_", y=metric, hue="params.user_input_logvar")
    plt.xlabel("Lambda")
    plt.ylabel(human_readable_name)
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.ylim(ylim)
    plt.tight_layout()
    plt.show()


def plot_model_info(runs):
    # Make a new DF exploding out each weight/bias metric into its own row; this is the format
    # seaborn wants
    non_metric_columns = [c for c in runs.columns if not c.startswith("metrics.")]
    df = runs.melt(id_vars=non_metric_columns, var_name="metric", value_name="value")

    # Rename columns from params.PARAM_NAME to just param_name
    df = df.rename(columns=lambda x: x.split(".")[-1].lower() if x.startswith("params.") else x)

    # Remove the "metrics." prefix from the metric names
    df["metric"] = df["metric"].str.replace("metrics.", "", regex=False)

    # Metric names will be things like "layer_0_weights_mean_std". Break this into parts.
    new_cols = df["metric"].str.extract(
        r"layer_(?P<layer>\d+)_(?P<param>\w+)_(?P<statistic>\w+)_(?P<population>\w+)$", expand=True
    )

    # Add the new columns to the dataframe
    df = pd.concat([df, new_cols], axis=1)

    # Drop NaN values; this removes all rows that don't match the regex in the .extract() call above
    df = df.dropna()

    # Make layer numeric so seaborn sorts the x-axis correctly
    df["layer"] = df["layer"].astype(int)

    lambda_order = sorted(df["lambda_"].unique(), key=float)
    for name, group in df.groupby("param"):
        sns.lineplot(
            group[(group["statistic"] == "logvar") & (group["population"] == "mean")],
            x="layer",
            y="value",
            hue="lambda_",
            hue_order = lambda_order,
            palette="crest",
        )
        plt.ylabel(f"SNN {name} log variance")
        plt.tight_layout()
        plt.show()



def main():
    mlflow.set_tracking_uri("/data/projects/SVAE/mlruns")
    df = search_runs_by_params(
        experiment_name="LitSVAE_RDL",
        params={
            "latent_dim": 5,
            "decoder_source": "aee20ec5d660489eae8f5c85832362ed",
            "learning_rate": 1e-3,
            "decoder_pixel_covariance": str(PixelCovariance.ISOTROPIC),
            "epochs": 1000,
        },
        finished_only=True,
    )
    df["lambda_f"] = df["params.lambda_"].astype(float)
    df = df.sort_values("lambda_f")

    plot_model_info(df)
    plot_inference_goodness(df)
    plot_metric(
        df,
        "metrics.test_reconstruction",
        "Average Reconstruction Log Likelihood",
        ylim=(2000, None),
    )
    plot_metric(df, "metrics.test_kl", "Average KL(q(z|x)||p(z))")
    plot_metric(df[~np.isinf(df["lambda_f"])], "metrics.test_fisher_information_matrix", "FIM term")
    plot_metric(df[~np.isinf(df["lambda_f"])], "metrics.test_entropy", "Entropy term")



if __name__ == "__main__":
    main()
