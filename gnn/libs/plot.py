import os
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator


def plot_metric_at_k(
    tr_loss: List[float],
    val_loss: List[float],
    tr_acc: List[float],
    val_acc: List[float],
    parent_save_path: str,
) -> None:
    sns.set_style("darkgrid")

    epochs = len(tr_loss)
    # plot tr/val loss
    loss_df = pd.DataFrame(
        {
            "value": tr_loss + val_loss,
            "data": ["train"] * len(tr_loss) + ["val"] * len(val_loss),
            "epochs": [i for i in range(epochs)] * 2,
        }
    )
    plot_metric(
        df=loss_df,
        metric_name="loss",
        save_path=os.path.join(parent_save_path, "loss.png"),
        hue="data",
    )

    # plot tr/val accuracy
    acc_df = pd.DataFrame(
        {
            "value": tr_acc + val_acc,
            "data": ["train"] * len(tr_loss) + ["val"] * len(val_loss),
            "epochs": [i for i in range(epochs)] * 2,
        }
    )
    plot_metric(
        df=acc_df,
        metric_name="accuracy",
        save_path=os.path.join(parent_save_path, "acc.png"),
        hue="data",
    )


def plot_metric(
    df: pd.DataFrame,
    metric_name: str,
    save_path: str,
    hue: Optional[str],
) -> None:
    if hue is not None:
        sns.lineplot(x="epochs", y="value", data=df, hue=hue, marker="o")
        title = f"{metric_name} at every epoch"
    else:
        sns.lineplot(x="epochs", y="value", data=df, marker="o")
        title = f"{metric_name} at every epoch"
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(5))
    plt.ylabel(metric_name)
    plt.title(title)
    plt.savefig(save_path)
    plt.show()
    plt.close()
