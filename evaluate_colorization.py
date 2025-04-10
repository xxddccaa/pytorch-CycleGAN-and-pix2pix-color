import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import lpips
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    return ssim(img1, img2, channel_axis=2, data_range=255)

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    return psnr(img1, img2, data_range=255)

def calculate_mse(img1, img2):
    """Calculate MSE between two images"""
    return mse(img1, img2)

def calculate_mae(img1, img2):
    """Calculate Mean Absolute Error between two images"""
    return np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))

def rgb2lab(rgb_image):
    """Convert RGB image to LAB color space"""
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    return lab_image

def calculate_color_accuracy(img1, img2):
    """Calculate color accuracy in ab channels of LAB color space"""
    lab1 = rgb2lab(img1)
    lab2 = rgb2lab(img2)
    # Extract a and b channels
    ab1 = lab1[:, :, 1:3]
    ab2 = lab2[:, :, 1:3]
    # Calculate mean absolute error for color channels
    color_error = np.mean(np.abs(ab1.astype(np.float32) - ab2.astype(np.float32)))
    return color_error

class LPIPS_Metric:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = lpips.LPIPS(net='alex').to(device)
        
    def calculate_lpips(self, img1, img2):
        """Calculate LPIPS perceptual distance between two images"""
        # Convert from numpy to torch tensor
        transform = transforms.ToTensor()
        img1_tensor = transform(img1).unsqueeze(0).to(self.device)
        img2_tensor = transform(img2).unsqueeze(0).to(self.device)
        
        # Normalize to [-1, 1]
        img1_tensor = img1_tensor * 2 - 1
        img2_tensor = img2_tensor * 2 - 1
        
        with torch.no_grad():
            distance = self.model(img1_tensor, img2_tensor)
        return distance.item()

class FID_Metric:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = models.inception_v3(pretrained=True).to(device)
        self.model.eval()
        
        # Remove classifier
        self.model.fc = torch.nn.Identity()
        
        # Set transforms
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_features(self, images_list):
        """Extract features from a list of image paths"""
        features = []
        
        for img_path in tqdm(images_list, desc="Extracting features"):
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feat = self.model(img)
            
            features.append(feat.cpu().numpy())
            
        return np.concatenate(features, axis=0)
    
    def calculate_fid(self, real_images, fake_images):
        """Calculate Fréchet Inception Distance between two sets of images"""
        real_features = self.get_features(real_images)
        fake_features = self.get_features(fake_images)
        
        # Calculate mean and covariance
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
        
        # Calculate FID
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        return fid

def parse_image_name(path):
    """Extract the base name from file path for grouping images"""
    filename = os.path.basename(path)
    
    # 移除文件扩展名
    filename_no_ext = os.path.splitext(filename)[0]
    
    # 处理文件名中的后缀部分
    if '_real_A' in filename_no_ext:
        # 移除 _real_A 后缀
        base_name = filename_no_ext.replace('_real_A', '')
    elif '_real_B_rgb' in filename_no_ext:
        # 移除 _real_B_rgb 后缀
        base_name = filename_no_ext.replace('_real_B_rgb', '')
    elif '_fake_B_rgb' in filename_no_ext:
        # 移除 _fake_B_rgb 后缀
        base_name = filename_no_ext.replace('_fake_B_rgb', '')
    else:
        # 如果不包含预期的后缀，返回原始文件名
        base_name = filename_no_ext
        
    return base_name

def collect_image_paths(results_dir):
    """Collect all image paths and group them by base name"""
    all_images = glob.glob(os.path.join(results_dir, "*.png"))
    
    print(f"Total images found: {len(all_images)}")
    
    # 打印前几个文件名作为示例
    if len(all_images) > 0:
        print("Sample filenames:")
        for i, img_path in enumerate(all_images[:5]):
            print(f"  {i+1}. {os.path.basename(img_path)}")
    
    # Group images by base name
    image_groups = {}
    
    # 计数器用于调试
    real_A_count = 0
    real_B_count = 0
    fake_B_count = 0
    
    for img_path in all_images:
        base_name = parse_image_name(img_path)
        
        if base_name not in image_groups:
            image_groups[base_name] = {'real_A': None, 'real_B_rgb': None, 'fake_B_rgb': None}
        
        if '_real_A.png' in img_path:
            image_groups[base_name]['real_A'] = img_path
            real_A_count += 1
        elif '_real_B_rgb.png' in img_path:
            image_groups[base_name]['real_B_rgb'] = img_path
            real_B_count += 1
        elif '_fake_B_rgb.png' in img_path:
            image_groups[base_name]['fake_B_rgb'] = img_path
            fake_B_count += 1
    
    # 打印找到的各类型图像数量
    print(f"Found images - real_A: {real_A_count}, real_B_rgb: {real_B_count}, fake_B_rgb: {fake_B_count}")
    
    # 打印前几个基本名称
    if len(image_groups) > 0:
        print("Sample base names:")
        sample_keys = list(image_groups.keys())[:5]
        for i, base_name in enumerate(sample_keys):
            print(f"  {i+1}. {base_name}")
            group = image_groups[base_name]
            print(f"     real_A: {os.path.basename(group['real_A']) if group['real_A'] else 'None'}")
            print(f"     real_B_rgb: {os.path.basename(group['real_B_rgb']) if group['real_B_rgb'] else 'None'}")
            print(f"     fake_B_rgb: {os.path.basename(group['fake_B_rgb']) if group['fake_B_rgb'] else 'None'}")
    
    # Filter out incomplete groups
    complete_groups = {k: v for k, v in image_groups.items() 
                       if v['real_A'] and v['real_B_rgb'] and v['fake_B_rgb']}
    
    # 打印完整组和不完整组的数量
    incomplete_count = len(image_groups) - len(complete_groups)
    print(f"Complete image groups: {len(complete_groups)}")
    print(f"Incomplete image groups: {incomplete_count}")
    
    if incomplete_count > 0 and len(image_groups) > 0:
        print("Sample incomplete groups:")
        incomplete_groups = {k: v for k, v in image_groups.items() 
                            if not (v['real_A'] and v['real_B_rgb'] and v['fake_B_rgb'])}
        sample_keys = list(incomplete_groups.keys())[:5]
        for i, base_name in enumerate(sample_keys):
            print(f"  {i+1}. {base_name}")
            group = incomplete_groups[base_name]
            print(f"     real_A: {os.path.basename(group['real_A']) if group['real_A'] else 'None'}")
            print(f"     real_B_rgb: {os.path.basename(group['real_B_rgb']) if group['real_B_rgb'] else 'None'}")
            print(f"     fake_B_rgb: {os.path.basename(group['fake_B_rgb']) if group['fake_B_rgb'] else 'None'}")
    
    return complete_groups

def evaluate_images(image_groups, metrics_to_compute=None, use_lpips=True, use_fid=False):
    """Evaluate the model performance on the provided image groups"""
    if metrics_to_compute is None:
        metrics_to_compute = ['ssim', 'psnr', 'mse', 'mae', 'color_error']
    
    # Initialize metrics
    metrics = {metric: [] for metric in metrics_to_compute}
    
    if use_lpips:
        try:
            lpips_metric = LPIPS_Metric()
            metrics['lpips'] = []
        except Exception as e:
            print(f"LPIPS initialization failed: {e}")
            print("Continuing without LPIPS metric...")
            use_lpips = False
    
    # 检查是否有图像组可处理
    if not image_groups:
        print("No complete image groups found. Cannot calculate metrics.")
        return metrics
    
    # Process each image group
    for name, paths in tqdm(image_groups.items(), desc="Evaluating images"):
        try:
            # Load images
            real_B = cv2.imread(paths['real_B_rgb'])
            fake_B = cv2.imread(paths['fake_B_rgb'])
            
            # 确保图像已成功加载
            if real_B is None or fake_B is None:
                print(f"Warning: Could not load images for {name}")
                print(f"  real_B path: {paths['real_B_rgb']}")
                print(f"  fake_B path: {paths['fake_B_rgb']}")
                continue
            
            # Convert BGR to RGB
            real_B = cv2.cvtColor(real_B, cv2.COLOR_BGR2RGB)
            fake_B = cv2.cvtColor(fake_B, cv2.COLOR_BGR2RGB)
            
            # 确保两幅图像尺寸相同
            if real_B.shape != fake_B.shape:
                print(f"Warning: Image shapes don't match for {name}")
                print(f"  real_B shape: {real_B.shape}")
                print(f"  fake_B shape: {fake_B.shape}")
                continue
            
            # Calculate metrics
            if 'ssim' in metrics_to_compute:
                metrics['ssim'].append(calculate_ssim(real_B, fake_B))
            if 'psnr' in metrics_to_compute:
                metrics['psnr'].append(calculate_psnr(real_B, fake_B))
            if 'mse' in metrics_to_compute:
                metrics['mse'].append(calculate_mse(real_B, fake_B))
            if 'mae' in metrics_to_compute:
                metrics['mae'].append(calculate_mae(real_B, fake_B))
            if 'color_error' in metrics_to_compute:
                metrics['color_error'].append(calculate_color_accuracy(real_B, fake_B))
            if use_lpips:
                try:
                    metrics['lpips'].append(lpips_metric.calculate_lpips(real_B, fake_B))
                except Exception as e:
                    print(f"LPIPS calculation failed for {name}: {e}")
        except Exception as e:
            print(f"Error processing image group {name}: {e}")
    
    # 检查是否成功计算了任何指标
    for metric_name, values in metrics.items():
        print(f"Calculated {metric_name} for {len(values)} images")
    
    # Calculate FID if requested
    if use_fid and image_groups:
        try:
            real_images = [paths['real_B_rgb'] for paths in image_groups.values()]
            fake_images = [paths['fake_B_rgb'] for paths in image_groups.values()]
            
            fid_metric = FID_Metric()
            fid_score = fid_metric.calculate_fid(real_images, fake_images)
            metrics['fid'] = fid_score
            print(f"FID score: {fid_score}")
        except Exception as e:
            print(f"FID calculation failed: {e}")
    
    return metrics

def summarize_metrics(metrics):
    """Summarize the metrics and return a DataFrame"""
    summary = {}
    
    # 检查是否有数据可供计算
    has_data = False
    for metric_name, values in metrics.items():
        if metric_name != 'fid' and len(values) > 0:
            has_data = True
            break
    
    if not has_data:
        print("No metrics data available to summarize.")
        # 返回一个空的DataFrame而不是引发错误
        return pd.DataFrame()
    
    # Calculate statistics for each metric
    for metric_name, values in metrics.items():
        if metric_name == 'fid':  # FID is a single value
            if values is not None:
                summary[metric_name] = values
        else:
            if len(values) > 0:
                summary[f"{metric_name}_mean"] = np.mean(values)
                summary[f"{metric_name}_std"] = np.std(values)
                summary[f"{metric_name}_min"] = np.min(values)
                summary[f"{metric_name}_max"] = np.max(values)
    
    return pd.DataFrame([summary])

def plot_metrics_distribution(metrics, output_dir):
    """Plot the distribution of metrics and save to output directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查是否有数据可供绘图
    has_data_to_plot = False
    
    for metric_name, values in metrics.items():
        if metric_name == 'fid':  # Skip FID as it's a single value
            continue
            
        if len(values) == 0:
            print(f"No data available for {metric_name} distribution plot.")
            continue
            
        has_data_to_plot = True
        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=30, alpha=0.7)
        plt.title(f'Distribution of {metric_name.upper()}')
        plt.xlabel(metric_name.upper())
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'{metric_name}_distribution.png'))
        plt.close()
    
    if not has_data_to_plot:
        print("No distribution plots generated due to lack of data.")

def main():
    parser = argparse.ArgumentParser(description='Evaluate image colorization results')
    parser.add_argument('--results_dir', type=str, default='./results/tongyong_l2ab_4/testA_35/images',
                      help='directory containing result images')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                      help='directory to save evaluation results')
    parser.add_argument('--use_lpips', action='store_true', default=True,
                      help='whether to compute LPIPS metric (slower)')
    parser.add_argument('--use_fid', action='store_true', default=False,
                      help='whether to compute FID metric (much slower)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect image paths
    print("Collecting image paths...")
    image_groups = collect_image_paths(args.results_dir)
    print(f"Found {len(image_groups)} complete image groups")
    
    # Evaluate images
    if len(image_groups) > 0:
        print("Evaluating images...")
        metrics = evaluate_images(image_groups, use_lpips=args.use_lpips, use_fid=args.use_fid)
        
        # Summarize metrics
        summary = summarize_metrics(metrics)
        
        # 检查是否计算了任何指标
        if not summary.empty:
            # Print summary
            print("\nEvaluation Results:")
            print(summary.to_string(index=False))
            
            # Save results
            summary.to_csv(os.path.join(args.output_dir, 'metrics_summary.csv'), index=False)
            
            # Save full metrics
            results_df = pd.DataFrame(metrics)
            results_df.to_csv(os.path.join(args.output_dir, 'metrics_full.csv'), index=False)
            
            # Plot metrics distribution
            plot_metrics_distribution(metrics, args.output_dir)
            
            print(f"\nResults saved to {args.output_dir}")
        else:
            print("No metrics could be calculated. Check the image files and paths.")
    else:
        print("No complete image groups found. Cannot continue with evaluation.")
        print("Please check that your image files follow the naming convention:")
        print("  - base_name_real_A.png")
        print("  - base_name_real_B_rgb.png")
        print("  - base_name_fake_B_rgb.png")

if __name__ == "__main__":
    main()