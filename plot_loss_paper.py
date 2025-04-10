import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter1d
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

# Set font settings for publication-quality figures
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']  # Add SimSun for Chinese
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False  # Correctly display negative symbols with Chinese
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Path to loss log file
log_file = "checkpoints/tongyong_l2ab_4/loss_log.txt"

# Regular expression pattern to extract loss values
pattern = r'\(epoch: (\d+), iters: (\d+), time: [^)]+\) G_GAN: ([0-9.]+) G_L1: ([0-9.]+) D_real: ([0-9.]+) D_fake: ([0-9.]+)'

# Initialize dictionaries to store loss values by epoch
epoch_data = {}

# Maximum epoch to plot
max_epoch = 35

print(f"Reading loss log from: {log_file}")
try:
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                iter_num = int(match.group(2))
                g_gan = float(match.group(3))
                g_l1 = float(match.group(4))
                d_real = float(match.group(5))
                d_fake = float(match.group(6))
                
                # Skip if epoch is beyond what we want to plot
                if epoch > max_epoch:
                    continue
                    
                # Store losses by epoch and iteration
                if epoch not in epoch_data:
                    epoch_data[epoch] = {
                        'G_GAN': [], 'G_L1': [], 'D_real': [], 'D_fake': []
                    }
                    
                epoch_data[epoch]['G_GAN'].append(g_gan)
                epoch_data[epoch]['G_L1'].append(g_l1)
                epoch_data[epoch]['D_real'].append(d_real)
                epoch_data[epoch]['D_fake'].append(d_fake)
    
    # Calculate average loss per epoch
    epochs = sorted(epoch_data.keys())
    avg_losses = {
        'G_GAN': [], 'G_L1': [], 'D_real': [], 'D_fake': []
    }
    
    for epoch in epochs:
        for loss_type in avg_losses:
            avg_losses[loss_type].append(np.mean(epoch_data[epoch][loss_type]))
    
    # Apply smoothing
    sigma = 1.0  # Smoothing factor
    smoothed_losses = {}
    for loss_type, values in avg_losses.items():
        if len(values) > 3:  # Only smooth if we have enough data points
            smoothed_losses[loss_type] = gaussian_filter1d(values, sigma)
        else:
            smoothed_losses[loss_type] = values
    
    # Create better-organized figure for the 3 loss curves (excluding G_GAN)
    fig = plt.figure(figsize=(10, 8))
    
    # Create GridSpec with 1x3 layout instead of 2x2
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1], hspace=0.3)
    
    # Create subplots using GridSpec for the remaining 3 loss types
    ax1 = fig.add_subplot(gs[0, 0])  # G_L1
    ax2 = fig.add_subplot(gs[1, 0])  # D_real
    ax3 = fig.add_subplot(gs[2, 0])  # D_fake
    
    # Define colors and markers
    colors = ['#ff7f0e', '#2ca02c', '#d62728']  # Removed the G_GAN color
    markers = ['s', '^', 'D']  # Removed the G_GAN marker
    
    # Plot individual losses (without G_GAN)
    # G_L1
    ax1.plot(epochs, smoothed_losses['G_L1'], '-', 
             color=colors[0], marker=markers[0], 
             markevery=max(1, len(epochs)//8), 
             linewidth=1.5, markersize=4, 
             label='G_L1')
    ax1.set_title('生成器 L1 损失 (G_L1)')
    ax1.set_xlabel('训练轮次 (Epoch)')
    ax1.set_ylabel('损失值 (Loss)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # D_real
    ax2.plot(epochs, smoothed_losses['D_real'], '-', 
             color=colors[1], marker=markers[1], 
             markevery=max(1, len(epochs)//8), 
             linewidth=1.5, markersize=4, 
             label='D_real')
    ax2.set_title('判别器真实样本损失 (D_real)')
    ax2.set_xlabel('训练轮次 (Epoch)')
    ax2.set_ylabel('损失值 (Loss)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # D_fake
    ax3.plot(epochs, smoothed_losses['D_fake'], '-', 
             color=colors[2], marker=markers[2], 
             markevery=max(1, len(epochs)//8), 
             linewidth=1.5, markersize=4, 
             label='D_fake')
    ax3.set_title('判别器生成样本损失 (D_fake)')
    ax3.set_xlabel('训练轮次 (Epoch)')
    ax3.set_ylabel('损失值 (Loss)')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Set a main title
    plt.suptitle('黑白图像上色模型的训练损失曲线\nTraining Loss Curves for L*AB Colorization GAN', fontsize=14)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Save high-resolution figures for publication
    plt.savefig('figures/colorization_loss_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/colorization_loss_curves.pdf', bbox_inches='tight')
    
    print(f"Loss curves saved to figures/colorization_loss_curves.png and figures/colorization_loss_curves.pdf")
    
    # Also create a combined plot with the 3 metrics (excluding G_GAN) for comparison
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, smoothed_losses['G_L1'], '-', color=colors[0], marker=markers[0], 
             markevery=max(1, len(epochs)//8), linewidth=2, label='G_L1')
    plt.plot(epochs, smoothed_losses['D_real'], '-', color=colors[1], marker=markers[1], 
             markevery=max(1, len(epochs)//8), linewidth=2, label='D_real')
    plt.plot(epochs, smoothed_losses['D_fake'], '-', color=colors[2], marker=markers[2], 
             markevery=max(1, len(epochs)//8), linewidth=2, label='D_fake')
    
    plt.title('训练损失对比曲线 (Combined Loss Curves)', fontsize=14)
    plt.xlabel('训练轮次 (Epoch)', fontsize=12)
    plt.ylabel('损失值 (Loss)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, frameon=True, fancybox=True, framealpha=0.8, loc='upper right')
    plt.tight_layout()
    
    # Save the combined plot
    plt.savefig('figures/colorization_combined_loss.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/colorization_combined_loss.pdf', bbox_inches='tight')
    
    print(f"Combined loss curve saved to figures/colorization_combined_loss.png and figures/colorization_combined_loss.pdf")
    
    # Display the plots
    plt.show()

except Exception as e:
    print(f"Error processing the loss log file: {e}")
    # Create a test plot with synthetic data
    print("Creating a sample plot with synthetic data for demonstration...")
    
    # Generate synthetic data
    epochs = list(range(1, 36))
    synthetic_g_gan = 0.5 + 0.3 * np.exp(-np.array(epochs) / 10)
    synthetic_g_l1 = 10.0 - 5.0 * np.exp(-np.array(epochs) / 15)
    synthetic_d_real = 0.7 - 0.2 * np.exp(-np.array(epochs) / 8)
    synthetic_d_fake = 0.6 + 0.1 * np.exp(-np.array(epochs) / 12)
    
    # Add noise
    np.random.seed(42)
    synthetic_g_gan += np.random.normal(0, 0.05, len(epochs))
    synthetic_g_l1 += np.random.normal(0, 0.3, len(epochs))
    synthetic_d_real += np.random.normal(0, 0.03, len(epochs))
    synthetic_d_fake += np.random.normal(0, 0.03, len(epochs))
    
    # Apply smoothing
    sigma = 1.0
    synthetic_g_gan_smooth = gaussian_filter1d(synthetic_g_gan, sigma)
    synthetic_g_l1_smooth = gaussian_filter1d(synthetic_g_l1, sigma)
    synthetic_d_real_smooth = gaussian_filter1d(synthetic_d_real, sigma)
    synthetic_d_fake_smooth = gaussian_filter1d(synthetic_d_fake, sigma)
    
    # Create better-organized figure for the 3 loss curves (excluding G_GAN)
    fig = plt.figure(figsize=(10, 8))
    
    # Create GridSpec with 1x3 layout instead of 2x2
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1], hspace=0.3)
    
    # Create subplots using GridSpec for the remaining 3 loss types
    ax1 = fig.add_subplot(gs[0, 0])  # G_L1
    ax2 = fig.add_subplot(gs[1, 0])  # D_real
    ax3 = fig.add_subplot(gs[2, 0])  # D_fake
    
    # Define colors and markers
    colors = ['#ff7f0e', '#2ca02c', '#d62728']  # Removed the G_GAN color
    markers = ['s', '^', 'D']  # Removed the G_GAN marker
    
    # Plot individual losses with synthetic data (without G_GAN)
    # G_L1
    ax1.plot(epochs, synthetic_g_l1_smooth, '-', 
             color=colors[0], marker=markers[0], 
             markevery=max(1, len(epochs)//8), 
             linewidth=1.5, markersize=4, 
             label='G_L1')
    ax1.set_title('生成器 L1 损失 (G_L1)')
    ax1.set_xlabel('训练轮次 (Epoch)')
    ax1.set_ylabel('损失值 (Loss)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # D_real
    ax2.plot(epochs, synthetic_d_real_smooth, '-', 
             color=colors[1], marker=markers[1], 
             markevery=max(1, len(epochs)//8), 
             linewidth=1.5, markersize=4, 
             label='D_real')
    ax2.set_title('判别器真实样本损失 (D_real)')
    ax2.set_xlabel('训练轮次 (Epoch)')
    ax2.set_ylabel('损失值 (Loss)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # D_fake
    ax3.plot(epochs, synthetic_d_fake_smooth, '-', 
             color=colors[2], marker=markers[2], 
             markevery=max(1, len(epochs)//8), 
             linewidth=1.5, markersize=4, 
             label='D_fake')
    ax3.set_title('判别器生成样本损失 (D_fake)')
    ax3.set_xlabel('训练轮次 (Epoch)')
    ax3.set_ylabel('损失值 (Loss)')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Set a main title
    plt.suptitle('黑白图像上色模型的训练损失曲线（示例数据）\nTraining Loss Curves for L*AB Colorization GAN (Demo Data)', fontsize=14)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Save high-resolution figures for publication
    plt.savefig('figures/demo_colorization_loss_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/demo_colorization_loss_curves.pdf', bbox_inches='tight')
    
    print(f"Demo loss curves saved to figures/demo_colorization_loss_curves.png and figures/demo_colorization_loss_curves.pdf")
    
    # Also create a combined plot with the 3 metrics (excluding G_GAN) for comparison
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, synthetic_g_l1_smooth, '-', color=colors[0], marker=markers[0], 
             markevery=max(1, len(epochs)//8), linewidth=2, label='G_L1')
    plt.plot(epochs, synthetic_d_real_smooth, '-', color=colors[1], marker=markers[1], 
             markevery=max(1, len(epochs)//8), linewidth=2, label='D_real')
    plt.plot(epochs, synthetic_d_fake_smooth, '-', color=colors[2], marker=markers[2], 
             markevery=max(1, len(epochs)//8), linewidth=2, label='D_fake')
    
    plt.title('训练损失对比曲线（示例数据）(Combined Loss Curves - Demo)', fontsize=14)
    plt.xlabel('训练轮次 (Epoch)', fontsize=12)
    plt.ylabel('损失值 (Loss)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, frameon=True, fancybox=True, framealpha=0.8, loc='upper right')
    plt.tight_layout()
    
    # Save the combined plot
    plt.savefig('figures/demo_colorization_combined_loss.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/demo_colorization_combined_loss.pdf', bbox_inches='tight')
    
    print(f"Combined demo loss curve saved to figures/demo_colorization_combined_loss.png and figures/demo_colorization_combined_loss.pdf") 