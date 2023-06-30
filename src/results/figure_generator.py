from pathlib import Path
from typing import Optional

import pandas
import seaborn as sns
from matplotlib import pyplot as plt, ticker

sns.set_theme()


def generate_figure(csv: str, x: str, x_label: str, y_label: str, out: Optional[str] = None):
    data = pandas.read_csv(csv, delimiter=";")

    # Create a square figure
    plt.figure(figsize=(6, 6))

    for column in data.columns[1:]:
        if column.endswith("MIN") or column.endswith("MAX"):
            continue

        sns.lineplot(data=data, x=x, y=column, label=f"{column}")

    # x-axis ticks as integers.
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[::-1], labels=labels[::-1])

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if out is not None:
        plt.savefig(out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--csv", type=str, help="CSV filename.")
    parser.add_argument("-x", type=str, default="Step", help="The column name for the x axis variable.")
    parser.add_argument("--x-label", type=str, default="Epoch", help="The label for the x-axis.")
    parser.add_argument("--y-label", type=str, default="Mean foreground Dice", help="The label for the y-axis.")
    parser.add_argument(
        "-o", "--out", type=str, help="Output file name. If not provided, the figure will not be saved."
    )
    args = parser.parse_args()

    if not args.csv.endswith(".csv"):
        args.csv += ".csv"

    results_root = Path("/Users/joshstein/Documents/thesis/results/mnms")

    generate_figure(results_root / args.csv, args.x, args.x_label, args.y_label)
    plt.show()
