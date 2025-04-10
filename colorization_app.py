import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr
import argparse

# 确保能够导入项目的其他模块
sys.path.append('.')

from options.test_options import TestOptions
from models import create_model
from skimage import color


class ColorizeModel:
    def __init__(self, model_path, name, epoch='latest', cuda=True):
        """初始化着色模型"""
        self.opt = self._prepare_options(model_path, name, epoch, cuda)
        self.model = create_model(self.opt)
        self.model.setup(self.opt)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        print(f"Model {name} loaded successfully!")
    
    def _prepare_options(self, model_path, name, epoch, cuda):
        """准备模型选项"""
        # 创建一个临时的sys.argv保存原来的参数
        old_argv = sys.argv
        
        # 设置测试参数
        sys.argv = [
            'test.py',
            '--dataroot', './',  # 这个值在这里不重要，因为我们不用它加载数据
            '--name', name,
            '--model', 'colorization',
            '--netG', 'unet_256',
            '--dataset_mode', 'colorization',
            '--norm', 'instance',
            '--input_nc', '1',
            '--output_nc', '2',
            '--checkpoints_dir', model_path,
            '--epoch', epoch,
            '--eval',
        ]
        
        if not cuda:
            sys.argv.append('--gpu_ids')
            sys.argv.append('-1')
        
        # 加载选项
        opt = TestOptions().parse()
        
        # 恢复原来的sys.argv
        sys.argv = old_argv
        
        # 硬编码一些测试需要的参数
        opt.serial_batches = True  # 禁用数据洗牌
        opt.no_flip = True        # 禁用图像翻转
        opt.display_id = -1       # 无visdom显示
        opt.num_threads = 0       # 单线程加载数据
        opt.batch_size = 1        # 单张图像处理
        
        return opt
    
    def process_image(self, image):
        """处理上传的图像：转换为L通道，然后着色"""
        # 确保图像是RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 获取RGB格式的numpy数组
        rgb_img = np.array(image)
        
        # 保存原始图像尺寸以便后续恢复
        original_height, original_width = rgb_img.shape[:2]
        
        # 对图像尺寸进行调整，确保是256的倍数
        # UNet架构需要图像尺寸是2的幂次方或特定尺寸的倍数才能正常工作
        target_size = 256  # 因为我们使用的是unet_256
        new_width = (original_width // target_size) * target_size
        new_height = (original_height // target_size) * target_size
        
        # 确保尺寸至少为256x256
        new_width = max(target_size, new_width)
        new_height = max(target_size, new_height)
        
        # 调整图像尺寸
        if new_width != original_width or new_height != original_height:
            rgb_img = cv2.resize(rgb_img, (new_width, new_height))
        
        # 转换为Lab格式并提取L通道
        lab_img = color.rgb2lab(rgb_img)
        l_channel = lab_img[:, :, 0]
        
        # 归一化L通道
        l_normalized = (l_channel / 50.0) - 1.0
        
        # 转换为PyTorch张量
        l_tensor = torch.from_numpy(l_normalized).float().unsqueeze(0).unsqueeze(0)
        
        # 准备模型输入 - 创建一个dummy B tensor以满足模型的要求
        # 在测试阶段，B不会被使用，但模型仍然期望它存在
        dummy_ab = torch.zeros((1, 2, l_tensor.shape[2], l_tensor.shape[3]), dtype=torch.float)
        data = {'A': l_tensor, 'B': dummy_ab, 'A_paths': 'uploaded_image', 'B_paths': 'dummy_path'}
        self.model.set_input(data)
        
        # 运行模型推理
        self.model.test()
        
        # 获取结果
        fake_ab = self.model.fake_B.detach().cpu().numpy()[0]
        # 反归一化 (-1, 1) -> (0, 1) -> 实际范围
        fake_ab = fake_ab * 110.0
        
        # 重建完整的Lab图像
        l_channel = l_channel.reshape(l_channel.shape[0], l_channel.shape[1], 1)
        fake_ab = np.transpose(fake_ab, (1, 2, 0))
        
        # 确保尺寸匹配
        if fake_ab.shape[0] != l_channel.shape[0] or fake_ab.shape[1] != l_channel.shape[1]:
            fake_ab = cv2.resize(fake_ab, (l_channel.shape[1], l_channel.shape[0]))
        
        # 合并L通道和生成的ab通道
        colorized_lab = np.concatenate([l_channel, fake_ab], axis=2)
        
        # 转换回RGB
        colorized_rgb = color.lab2rgb(colorized_lab) * 255
        colorized_rgb = colorized_rgb.astype(np.uint8)
        
        # 恢复原始图像尺寸
        if new_width != original_width or new_height != original_height:
            colorized_rgb = cv2.resize(colorized_rgb, (original_width, original_height))
            rgb_img = cv2.resize(rgb_img, (original_width, original_height))
        
        # 创建灰度图像以显示
        grayscale_img = np.repeat(l_channel, 3, axis=2)
        grayscale_img = ((grayscale_img + 100) / 200 * 255).astype(np.uint8)
        
        # 如果灰度图尺寸需要调整
        if new_width != original_width or new_height != original_height:
            grayscale_img = cv2.resize(grayscale_img, (original_width, original_height))
        
        return rgb_img, grayscale_img, colorized_rgb
    
    def colorize(self, input_image):
        """用于Gradio界面的着色函数"""
        if input_image is None:
            return None, None, None
        
        # 如果输入是文件路径，则打开图像
        if isinstance(input_image, str):
            input_image = Image.open(input_image).convert('RGB')
        elif isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        
        # 处理图像
        original, grayscale, colorized = self.process_image(input_image)
        
        return original, grayscale, colorized


def create_gradio_interface(model):
    """创建Gradio界面"""
    with gr.Blocks(title="图像着色应用") as app:
        gr.Markdown("# 图像着色应用")
        gr.Markdown("上传一张图片，系统将自动将其转为灰度图并进行着色")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="上传图片", type="pil")
                colorize_btn = gr.Button("开始着色", variant="primary")
            
        with gr.Row():
            with gr.Column(scale=1):
                original_image = gr.Image(label="原始图片")
            with gr.Column(scale=1):
                grayscale_image = gr.Image(label="灰度图片")
            with gr.Column(scale=1):
                colorized_image = gr.Image(label="着色结果")
        
        colorize_btn.click(
            fn=model.colorize,
            inputs=[input_image],
            outputs=[original_image, grayscale_image, colorized_image]
        )
        
        gr.Markdown("## 使用说明")
        gr.Markdown("1. 点击上方的图像区域上传图片或拖放图片")
        gr.Markdown("2. 点击'开始着色'按钮")
        gr.Markdown("3. 等待几秒钟，系统会显示原始图片、转换后的灰度图和着色结果")
        gr.Markdown("4. 如果对结果不满意，可以上传新图片重试")
        gr.Markdown("5. 注意：图像会被调整尺寸以适应模型的需求，然后在输出时恢复原始尺寸")
        
    return app


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='图像着色应用')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                        help='模型检查点目录的路径')
    parser.add_argument('--name', type=str, default='tongyong_l2ab_4',
                        help='模型名称')
    parser.add_argument('--epoch', type=str, default='35',
                        help='使用哪个epoch的模型，设置为latest使用最新模型')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用CPU而不是GPU')
    parser.add_argument('--port', type=int, default=7860,
                        help='Gradio应用的端口号')
    parser.add_argument('--share', action='store_true',
                        help='生成可分享的链接')
    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置模型路径和名称
    model_path = args.checkpoints_dir
    model_name = args.name
    model_epoch = args.epoch
    
    # 检测CUDA是否可用
    use_cuda = torch.cuda.is_available() and not args.cpu
    
    print(f"Loading colorization model: {model_name}, epoch: {model_epoch}")
    print(f"CUDA available: {use_cuda}")
    
    # 加载模型
    colorize_model = ColorizeModel(model_path, model_name, model_epoch, use_cuda)
    
    # 创建Gradio界面
    app = create_gradio_interface(colorize_model)
    
    # 启动应用
    app.launch(server_port=args.port, server_name="0.0.0.0", share=args.share)
    

if __name__ == "__main__":
    main()