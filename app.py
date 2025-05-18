import os
import sys
import numpy as np
import cv2
from PIL import Image
import gradio as gr
import requests
import tempfile
import io

# Ensure project modules can be imported
sys.path.append('.')

# API endpoint configuration - update this to your API server address
API_URL = "http://10.136.19.26:9700/colorize"

def colorize_image(input_image):
    """Send image to API for colorization"""
    if input_image is None:
        return None, None
    
    # Convert input to PIL Image if it's numpy array
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)
    
    # Prepare to save image to temporary file
    temp_file_path = None
    try:
        # Save image to temporary file
        temp_file_path = tempfile.mktemp(suffix='.png')
        input_image.save(temp_file_path, format='PNG')
        
        # Create grayscale version for display
        grayscale_img = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2GRAY)
        grayscale_img = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)
        
        # Send request to API
        with open(temp_file_path, 'rb') as f:
            files = {'file': ('image.png', f, 'image/png')}
            response = requests.post(API_URL, files=files, timeout=60)
        
        # Check for errors
        if response.status_code != 200:
            error_message = f"API request failed with status {response.status_code}"
            try:
                error_detail = response.json().get('detail', '')
                if error_detail:
                    error_message += f": {error_detail}"
            except:
                pass
            raise Exception(error_message)
        
        # Read the image from response content
        colorized_img = Image.open(io.BytesIO(response.content))
        colorized_img = np.array(colorized_img)
        
        # Ensure the image is in RGB format (not BGR)
        if colorized_img.shape[2] == 3:  # Has 3 channels
            colorized_img = cv2.cvtColor(colorized_img, cv2.COLOR_BGR2RGB)
        
        return grayscale_img, colorized_img
    
    except Exception as e:
        print(f"Error during colorization: {str(e)}")
        # Return grayscale image and an error message image
        if grayscale_img is not None:
            error_img = np.zeros_like(grayscale_img)
            # Add error text to the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(error_img, f"Error: {str(e)}", (10, 30), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            return grayscale_img, error_img
        return None, None
    
    finally:
        # Clean up temporary files
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def create_gradio_interface():
    """Create Gradio interface"""
    with gr.Blocks(title="图像着色应用") as app:
        gr.Markdown("# 图像着色应用")
        gr.Markdown("上传一张图片，系统将自动将其转为灰度图并进行着色")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="上传图片", type="pil")
                colorize_btn = gr.Button("开始着色", variant="primary")
            
        with gr.Row():
            with gr.Column(scale=1):
                grayscale_image = gr.Image(label="灰度图片")
            with gr.Column(scale=1):
                colorized_image = gr.Image(label="着色结果")
        
        colorize_btn.click(
            fn=colorize_image,
            inputs=[input_image],
            outputs=[grayscale_image, colorized_image]
        )
        
        gr.Markdown("## 使用说明")
        gr.Markdown("1. 点击上方的图像区域上传图片或拖放图片")
        gr.Markdown("2. 点击'开始着色'按钮")
        gr.Markdown("3. 等待几秒钟，系统会显示转换后的灰度图和着色结果")
        gr.Markdown("4. 如果长时间没有响应或出现错误，请检查API服务器是否正常运行")
        
    return app

def main():
    """Main function to launch the Gradio app"""
    app = create_gradio_interface()
    app.launch(server_name="0.0.0.0")

if __name__ == "__main__":
    main() 