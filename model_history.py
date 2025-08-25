import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_history(my_args):
    """
    Enhanced plotting function that:
    - Handles interrupted training
    - Better visualizes partial progress
    - Automatically adapts to available metrics
    """
    try:
        # Try loading both joblib and CSV history
        history_joblib = joblib.load(f"{my_args.model_file}.history")
        history_csv = pd.read_csv("training_log.csv")
        history = {**history_joblib, **history_csv.to_dict('list')}  # Merge
    except:
        try:
            history = joblib.load(f"{my_args.model_file}.history")
        except:
            history = pd.read_csv("training_log.csv").to_dict('list')

    if not history:
        print("No training history found!")
        return

    # Determine completed epochs
    epochs = len(history.get('loss', []))
    if epochs == 0:
        print("No training epochs completed")
        return

    # Set up plot styles
    metric_groups = {
        'loss': ['loss', 'val_loss'],
        'accuracy': ['binary_accuracy', 'val_binary_accuracy'],
        'auc': ['auc', 'val_auc', 'pr_auc', 'val_pr_auc']
    }

    # Create subplots for each metric group
    fig, axes = plt.subplots(len(metric_groups), 1, figsize=(10, 15))
    if len(metric_groups) == 1:
        axes = [axes]  # Ensure axes is always iterable

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    linestyles = ['-', '--', '-.', ':']
    
    for ax, (group_name, metrics) in zip(axes, metric_groups.items()):
        # Plot only available metrics
        for i, metric in enumerate(metrics):
            if metric in history:
                ax.plot(
                    range(epochs), 
                    history[metric][:epochs], 
                    label=metric.replace('_', ' ').title(),
                    color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)],
                    marker='o' if epochs < 20 else None
                )
        
        ax.set_title(group_name.title())
        ax.set_xlabel('Epoch')
        ax.set_ylabel(group_name.title())
        ax.legend()
        ax.grid(True)

        # Mark early stopping point if validation loss exists
        if 'val_loss' in history and epochs < len(history['val_loss']):
            ax.axvline(epochs-1, color='r', linestyle=':', 
                      label='Early Stop' if group_name == 'loss' else '')
    
    plt.tight_layout()
    
    # Save and clear
    learning_curve_filename = f"{my_args.model_file}.learning_curve.png"
    plt.savefig(learning_curve_filename, dpi=300)
    plt.close()
    
    print(f"Saved learning curves to {learning_curve_filename}")