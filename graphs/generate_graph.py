import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read your CSV file
df = pd.read_csv('graphs/summary_res.csv', sep=',')  

# Drop the "Data" and "novelty" columns
df = df.drop(columns=["Data", "novelty"], errors='ignore')

# Check the unique algorithms and sort by K for neat plotting
df = df.sort_values(by=["Algo", "K"])

# Set a consistent style and palette
sns.set_style("whitegrid")
algos = df["Algo"].unique()
palette = dict(zip(algos, sns.color_palette("tab10", len(algos))))

# Helper function to create line plots for a given metric
def plot_metric_line(metric, y_label):
    plt.figure(figsize=(7,5))

    sns.lineplot(
        data=df, 
        x="K", 
        y=metric, 
        hue="Algo", 
        style="Algo",
        markers=True, 
        dashes=False, 
        palette=palette,
        linewidth=2,
        markersize=8
    )

    plt.title(f"{y_label} by Algorithm", fontsize=14)
    plt.xlabel("K", fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    plt.yscale("log")
    
    max_value = df[metric].max()
    dynamic_max = max_value * 1.1
    rounded_max = round_to_nice_value(dynamic_max)

    fixed_ticks = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]

    y_ticks = [tick for tick in fixed_ticks if tick <= rounded_max]

    if rounded_max > fixed_ticks[-1]:
        y_ticks.append(rounded_max)

    plt.yticks(y_ticks, [f"{y:.3f}" for y in y_ticks], fontsize=10)
    
    plt.xticks([10, 25, 50], fontsize=10)
    
    plt.legend(
        title="Algorithm",
        fontsize=10,
        title_fontsize=12,
        loc='upper left',
        bbox_to_anchor=(1.05, 1)
    )
    plt.tight_layout()
    plt.show()

def round_to_nice_value(value):
    if value <= 0.05:
        return 0.05
    elif value <= 0.1:
        return 0.1
    elif value <= 0.15:
        return 0.15
    elif value <= 0.2:
        return 0.2
    else:
        return round(value * 10) / 10

plot_metric_line("Precision@k", "Precision@K")

plot_metric_line("Recall@k", "Recall@K")

plot_metric_line("NDCG@k", "NDCG@K")

plot_metric_line("Mean average precision", "Mean Average Precision")

def plot_diversity(y_label="Diversity by Algorithm"):
    plt.figure(figsize=(7, 5))
    
    sns.barplot(
        data=df, 
        x="K", 
        y="diversity", 
        hue="Algo", 
        palette=palette
    )
    
    plt.title(y_label, fontsize=14)
    plt.xlabel("K", fontsize=12)
    plt.ylabel("Diversity", fontsize=12)

    plt.legend(
        title="Algorithm",
        fontsize=10,
        title_fontsize=12,
        loc='upper left',
        bbox_to_anchor=(1.05, 1)
    )
    
    plt.xticks([0, 1, 2], [10, 25, 50], fontsize=10)
    
    plt.tight_layout()
    plt.show()

plot_diversity("Diversity by Algorithm")