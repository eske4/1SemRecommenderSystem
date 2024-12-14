import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # To manually create legend handles

# Read CSV
df = pd.read_csv('summary_res.csv', sep=',')

# Drop the "Data" and "novelty" columns
df = df.drop(columns=["Data", "novelty"], errors='ignore')

# Sort values for consistent ordering
df = df.sort_values(by=["Algo", "K"])

# Style and palette
sns.set_style("whitegrid")
algos = df["Algo"].unique()
palette = dict(zip(algos, sns.color_palette("tab10", len(algos))))  # Algo-color mapping

# Combined Precision and Recall Plot
def plot_precision_recall_combined():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

    # Common y-axis ticks
    y_ticks = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15]
    y_labels = [f"{y:.3f}" for y in y_ticks]
    x_ticks = [10, 25, 50]


    # Plot Precision@K
    sns.lineplot(
        data=df, x="K", y="Precision@k", hue="Algo", style="Algo",
        markers=True, dashes=False, palette=palette, linewidth=2, markersize=8, ax=axes[0],
        legend=False  # Disable individual legend
    )

    axes[0].set_title("Precision@K by Algorithm", fontsize=14)
    axes[0].set_xlabel("K", fontsize=12)
    axes[0].set_ylabel("Precision/Recall", fontsize=12)
    axes[0].set_yscale("log")
    axes[0].set_yticks(y_ticks)
    axes[0].set_yticklabels(y_labels)
    axes[0].set_xticks(x_ticks)

    # Plot Recall@K
    sns.lineplot(
        data=df, x="K", y="Recall@k", hue="Algo", style="Algo",
        markers=True, dashes=False, palette=palette, linewidth=2, markersize=8, ax=axes[1],
        legend=False  # Disable individual legend
    )
    axes[1].set_title("Recall@K by Algorithm", fontsize=14)
    axes[1].set_xlabel("K", fontsize=12)


    fig_false, ax = plt.subplots()
    # Plot Precision@K
    precision_plot = sns.lineplot(
        data=df, x="K", y="Precision@k", hue="Algo", style="Algo",
        markers=True, dashes=False, palette=palette, linewidth=2, markersize=8, ax=ax,
        legend=True  # Disable individual legend
    )

    # Extract legend handles and labels
    handles, labels = precision_plot.get_legend_handles_labels()

    # Add the combined legend **at the top center**
    fig.legend(
        handles=handles, labels=labels, title="Algorithm", title_fontsize=12, fontsize=10,
        loc="center", bbox_to_anchor=(0.5, 0.05), ncol=len(algos)
    )

    # Adjust layout
    plt.subplots_adjust(wspace=0.15, hspace=0.05)  # Adjust spacing between the graphs
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the legend at the top
    plt.show()

plot_precision_recall_combined()
