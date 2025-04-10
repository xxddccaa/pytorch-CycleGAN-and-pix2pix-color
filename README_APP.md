# 图像着色应用

这是一个基于PyTorch-CycleGAN-and-pix2pix的图像着色应用程序，它利用Gradio创建一个简单友好的Web界面，让用户可以上传图片并进行着色。

## 主要功能

1. 上传彩色图像
2. 自动转换为灰度图
3. 使用预训练的pix2pix着色模型恢复彩色
4. 在界面中同时展示原始图像、灰度图像和着色结果

## 安装要求

确保已安装以下依赖项：

```bash
pip install torch torchvision scikit-image opencv-python pillow gradio
```

此外，您需要确保已经正确安装了PyTorch-CycleGAN-and-pix2pix项目的所有依赖项。

## 如何使用

### 1. 准备预训练模型

确保您已经训练好了一个着色模型（或使用提供的预训练模型）。模型应该保存在`checkpoints`目录中，以便应用程序能够找到它。

### 2. 运行应用程序

基本用法：

```bash
python colorization_app.py
```

这将使用默认设置启动应用程序：
- 模型路径：`./checkpoints`
- 模型名称：`tongyong_l2ab_4`
- Epoch：`35`

### 3. 自定义设置

您可以通过命令行参数自定义应用程序的行为：

```bash
python colorization_app.py --checkpoints_dir [模型路径] --name [模型名称] --epoch [模型epoch] --port [端口号] --share
```

参数说明：
- `--checkpoints_dir`：模型检查点目录的路径，默认为`./checkpoints`
- `--name`：模型名称，默认为`tongyong_l2ab_4`
- `--epoch`：使用哪个epoch的模型，默认为`35`，设置为`latest`使用最新模型
- `--cpu`：强制使用CPU而不是GPU
- `--port`：Gradio应用的端口号，默认为`7860`
- `--share`：添加此选项可生成可分享的链接，便于在不同设备上访问

示例：

```bash
python colorization_app.py --name my_colorization_model --epoch latest --share
```

### 4. 使用Web界面

一旦应用程序运行，您将看到一个Web界面：

1. 点击"上传图片"区域选择图片或拖放图片
2. 点击"开始着色"按钮
3. 等待几秒钟，系统会显示原始图片、转换后的灰度图和着色结果
4. 如果对结果不满意，可以上传新图片重试

## 应用架构

应用程序主要由以下部分组成：

1. **ColorizeModel类**：负责加载模型和处理图像
   - 初始化模型
   - 将输入图像转换为L通道
   - 使用模型进行着色
   - 后处理，将结果转换回RGB

2. **Gradio界面**：提供用户友好的Web界面
   - 图像上传
   - 显示处理结果
   - 用户交互

3. **参数解析**：处理命令行参数，提供灵活配置

## 问题排查

如果遇到以下问题，尝试以下解决方案：

1. **模型加载错误**：确保模型路径和名称正确，并且模型文件存在
2. **CUDA错误**：添加`--cpu`参数强制使用CPU
3. **端口冲突**：使用`--port`参数更改端口
4. **图像无法上传**：检查图像格式，确保是常见格式如PNG、JPG等

## 延伸应用

1. **批量处理**：修改代码以支持批量处理多张图像
2. **集成其他模型**：修改代码以支持其他图像处理模型
3. **添加更多选项**：例如调整图像尺寸、应用滤镜等 