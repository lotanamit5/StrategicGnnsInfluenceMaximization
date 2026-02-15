import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot_results(csv_file, dataset_name):
    df = pd.read_csv(csv_file)
    
    # Setup plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"{dataset_name}", fontsize=16)
    
    # 1. Non-strategic (Benchmark) Accuracy
    ax_bm = axes[0, 2]
    ax_bm.set_title("non-strategic (benchmark)")
    sns.lineplot(data=df, x='portion_remaining', y='benchmark_acc', hue='condition',
                 marker='.', ax=ax_bm)
    ax_bm.set_ylabel("accuracy")
    ax_bm.set_xlabel("portion of nodes that remain connected")
    ax_bm.grid(True)
    
    # 2. Naive Accuracy
    ax_naive_acc = axes[0, 0]
    ax_naive_acc.set_title("naive")
    sns.lineplot(data=df, x='portion_remaining', y='naive_acc', hue='condition',
                 marker='.', ax=ax_naive_acc)
    ax_naive_acc.set_ylabel("accuracy")
    ax_naive_acc.set_xlabel("portion of nodes that remain connected")
    ax_naive_acc.grid(True)
    
    # 3. Robust Accuracy
    ax_robust_acc = axes[0, 1]
    ax_robust_acc.set_title("robust (ours)")
    sns.lineplot(data=df, x='portion_remaining', y='robust_acc', hue='condition',
                 marker='.', ax=ax_robust_acc)
    ax_robust_acc.set_ylabel("accuracy")
    ax_robust_acc.set_xlabel("portion of nodes that remain connected")
    ax_robust_acc.grid(True)
    
    # 4. Naive Moved/Crossed - reshape data for seaborn
    naive_mc = df.melt(id_vars=['portion_remaining', 'condition'],
                       value_vars=['naive_pct_moved', 'naive_pct_crossed'],
                       var_name='metric', value_name='value')
    naive_mc['label'] = naive_mc['condition'] + '-' + naive_mc['metric'].str.replace('naive_pct_', '')
    
    ax_naive_mc = axes[1, 0]
    sns.lineplot(data=naive_mc, x='portion_remaining', y='value', hue='condition',
                 style='metric', markers=['o', 'X'], ax=ax_naive_mc)
    ax_naive_mc.set_ylabel("% moved/crossed")
    ax_naive_mc.set_xlabel("portion of nodes that remain connected")
    ax_naive_mc.grid(True)
    
    # 5. Robust Moved/Crossed
    robust_mc = df.melt(id_vars=['portion_remaining', 'condition'],
                        value_vars=['robust_pct_moved', 'robust_pct_crossed'],
                        var_name='metric', value_name='value')
    robust_mc['label'] = robust_mc['condition'] + '-' + robust_mc['metric'].str.replace('robust_pct_', '')
    
    ax_robust_mc = axes[1, 1]
    sns.lineplot(data=robust_mc, x='portion_remaining', y='value', hue='condition',
                 style='metric', markers=['o', 'X'], ax=ax_robust_mc)
    ax_robust_mc.set_ylabel("% moved/crossed")
    ax_robust_mc.set_xlabel("portion of nodes that remain connected")
    ax_robust_mc.grid(True)
    
    # 6. Bottom right empty
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_file = f"plot_centrality_{dataset_name}.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    plot_results(args.csv, args.dataset)
