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


def main():
    mlflow.set_tracking_uri("/data/projects/SVAE/mlruns")
    df = search_runs_by_params(
        experiment_name="LitSVAE_RDL",
        params={
            "decoder_source": "ba002b451919474c807c5ed52766eb93",
            "learning_rate": 1e-3,
        },
        finished_only=True,
    )
    df["lambda_f"] = df["params.lambda_"].astype(float)
    df = df.sort_values("lambda_f")

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
