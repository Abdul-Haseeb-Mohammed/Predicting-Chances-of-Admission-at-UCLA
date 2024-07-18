import matplotlib.pyplot as plt
import os
import seaborn as sns

def scatter_plot(data,save_path=None):
    plt.figure(figsize=(15,8))
    sns.scatterplot(data=data, 
           x='GRE_Score', 
           y='TOEFL_Score', 
           hue='Admit_Chance')
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Pair plot saved to {os.path.abspath(save_path)}")
    else:
        plt.show()
 
def loss_curve(loss_values,save_path=None):
    # Plotting the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Loss', color='blue')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Pair plot saved to {os.path.abspath(save_path)}")
    else:
        plt.show()