# 图像着色评估工具

这个项目提供了两个脚本来评估图像着色模型的性能，计算多种图像质量指标。

## 脚本说明

1. **evaluate_colorization.py** - 完整版评估脚本，支持更多高级指标（如LPIPS和FID）
2. **evaluate_basic.py** - 简化版评估脚本，只计算基本指标，依赖库更少

## 支持的评估指标

两个脚本都支持以下基本指标：

- **SSIM (结构相似性指数)** - 衡量两幅图像在结构上的相似程度，范围为[0,1]，越高越好
- **PSNR (峰值信噪比)** - 衡量重建图像的保真度，单位为dB，越高越好
- **MSE (均方误差)** - 衡量像素级差异的平方，越低越好
- **MAE (平均绝对误差)** - 衡量像素级绝对差异，越低越好
- **Color Error (色彩误差)** - Lab色彩空间中ab通道的平均绝对误差，越低越好

完整版脚本还支持以下高级指标：

- **LPIPS (感知相似度)** - 基于深度特征的感知相似度度量，越低越好
- **FID (Fréchet Inception Distance)** - 衡量生成图像和真实图像在特征空间的分布差异，越低越好

## 安装依赖

对于基本版本：

```bash
pip install numpy opencv-python scikit-image tqdm pandas matplotlib pillow
```

对于完整版本（包括LPIPS和FID指标）：

```bash
pip install -r evaluation_requirements.txt
```

## 使用方法

### 基本版本

```bash
python evaluate_basic.py --results_dir ./results/tongyong_l2ab_4/testA_35/images --output_dir ./evaluation_basic_results
```

### 完整版本

```bash
# 只计算基本指标 + LPIPS (默认)
python evaluate_colorization.py --results_dir ./results/tongyong_l2ab_4/testA_35/images --output_dir ./evaluation_results

# 计算所有指标，包括FID (较慢)
python evaluate_colorization.py --results_dir ./results/tongyong_l2ab_4/testA_35/images --output_dir ./evaluation_results --use_fid
```

## 参数说明

- `--results_dir`: 包含测试结果图像的目录路径
- `--output_dir`: 保存评估结果的目录路径
- `--use_lpips`: 是否计算LPIPS指标（仅在完整版中有效，默认开启）
- `--use_fid`: 是否计算FID指标（仅在完整版中有效，默认关闭，因为计算较慢）

## 输出结果

脚本会生成以下输出：

1. **metrics_summary.csv** - 包含所有指标的统计摘要（均值、标准差、最小值、最大值）
2. **metrics_full.csv** - 包含每张图像的详细指标
3. **[metric]_distribution.png** - 每个指标的直方图分布图

## 注意事项

1. 脚本假设结果图像按照`real_A.png`、`real_B_rgb.png`和`fake_B_rgb.png`的命名约定
2. 计算LPIPS需要GPU以获得更好的性能，但也可以在CPU上运行
3. 计算FID需要大量内存，并且处理时间较长
4. 确保scikit-image版本>=0.18.0，以支持channel_axis参数

## 指标解释

1. **SSIM (结构相似性指数)**：
   - 范围：0-1，值越高表示图像结构相似度越高
   - 优秀值：>0.9
   - 参考值：自然图像处理中，SSIM>0.85通常被认为是高质量的

2. **PSNR (峰值信噪比)**：
   - 单位：dB (分贝)
   - 优秀值：>30dB
   - 参考值：图像压缩中，30-50dB通常被认为是高质量的；着色任务中，25-35dB通常是好的结果

3. **MSE/MAE (均方误差/平均绝对误差)**：
   - 越低越好
   - 这些是像素级误差度量

4. **Color Error (色彩误差)**：
   - 越低越好
   - 这是Lab色彩空间中ab通道的平均绝对误差，专门用于评估着色质量

5. **LPIPS (感知相似度)**：
   - 范围：0-1，值越低表示感知相似度越高
   - 这个指标更符合人类感知的相似性判断

6. **FID (Fréchet Inception Distance)**：
   - 通常在0-100+范围内，值越低越好
   - 通常认为<50是可接受的，<20是非常好的 