import re
import os
import csv

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
            if epoch_data[epoch][loss_type]:
                avg_losses[loss_type].append(sum(epoch_data[epoch][loss_type]) / len(epoch_data[epoch][loss_type]))
            else:
                avg_losses[loss_type].append(0)
    
    # Create output directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Write the data to CSV
    csv_file = 'results/loss_data.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['Epoch', 'G_GAN', 'G_L1', 'D_real', 'D_fake'])
        # Write data
        for i, epoch in enumerate(epochs):
            writer.writerow([
                epoch, 
                avg_losses['G_GAN'][i], 
                avg_losses['G_L1'][i], 
                avg_losses['D_real'][i], 
                avg_losses['D_fake'][i]
            ])
    
    print(f"Loss data saved to {csv_file}")
    print("You can now import this CSV file into Excel to create your plots.")
    
    # Also print the data to console for reference
    print("\nAverage loss values by epoch:")
    print("Epoch\tG_GAN\tG_L1\tD_real\tD_fake")
    for i, epoch in enumerate(epochs):
        print(f"{epoch}\t{avg_losses['G_GAN'][i]:.3f}\t{avg_losses['G_L1'][i]:.3f}\t{avg_losses['D_real'][i]:.3f}\t{avg_losses['D_fake'][i]:.3f}")
    
except Exception as e:
    print(f"Error processing the loss log file: {e}")
    
    # Create sample data if we can't read the real data
    print("Creating sample data for demonstration...")
    
    # Sample data
    sample_data = []
    for epoch in range(1, 36):
        # Create some sample data with a pattern
        g_gan = 0.5 + 0.3 * (1 - epoch / 35)
        g_l1 = 10.0 - 5.0 * (epoch / 35)
        d_real = 0.7 - 0.2 * (epoch / 35)
        d_fake = 0.6 + 0.1 * (epoch / 35)
        
        sample_data.append([epoch, g_gan, g_l1, d_real, d_fake])
    
    # Create output directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Write the sample data to CSV
    csv_file = 'results/sample_loss_data.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['Epoch', 'G_GAN', 'G_L1', 'D_real', 'D_fake'])
        # Write data
        for row in sample_data:
            writer.writerow(row)
    
    print(f"Sample loss data saved to {csv_file}")
    print("You can now import this CSV file into Excel to create your plots.") 