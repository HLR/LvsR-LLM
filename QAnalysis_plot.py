import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("QAnalysis_ratios_median.csv")
df["Improvement %"] = df["Improvement %"].str.replace("%", "").astype(float)
sns.set_theme(style="white")
fig, axes = plt.subplots(1, 3, figsize=(8, 4), sharex=True)
example_counts = [10, 30, 100]

palette = [sns.color_palette("mako")[1], sns.color_palette("mako")[3]]

for i, examples in enumerate(example_counts):
    ax = axes[i]
    subset_df = df[df["In-Context Examples"] == examples]
    ax = sns.barplot(
        data=subset_df,
        x="Model",
        y="Improvement %",
        hue="Dataset",
        dodge=True,
        errorbar=None,
        ax=ax,
        palette=palette
    )

    ax.set_title(f"ICL Examples: {examples}", fontsize=17, weight='bold', pad=10)
    if i==0:
        ax.set_ylabel("KER Improvement (%)", fontsize=15)
    else:
        ax.set_ylabel("")
    ax.set_ylim(-5, 40)
    ax.axhline(0, color='black', linewidth=1)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    #for container in ax.containers:
    #    ax.bar_label(container, fmt="%.1f", label_type="edge", padding=3)

    ax.legend_.remove()
    ax.set_xlabel("", fontsize=20, weight='bold')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, [label.replace("_"," ") for label in labels], loc='center', bbox_to_anchor=(0.5, -0.03), ncol=2, fontsize=16)
sns.despine(left=False, bottom=False)
plt.tight_layout()
plt.savefig("QAnalysis_Figure7_300DPI.png", bbox_inches='tight', dpi=300)

