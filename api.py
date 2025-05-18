import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import tempfile
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Ensure project modules can be imported
sys.path.append('.')

from options.test_options import TestOptions
from models import create_model
from skimage import color

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ColorizeModel:
    def __init__(self, model_path, name, epoch='latest', cuda=True):
        """Initialize the colorization model"""
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
        """Prepare model options"""
        old_argv = sys.argv
        sys.argv = [
            'test.py',
            '--dataroot', './',
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
        
        opt = TestOptions().parse()
        sys.argv = old_argv
        
        opt.serial_batches = True
        opt.no_flip = True
        opt.display_id = -1
        opt.num_threads = 0
        opt.batch_size = 1
        
        return opt
    
    def process_image(self, image):
        """Process the uploaded image: convert to L channel and colorize"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        rgb_img = np.array(image)
        original_height, original_width = rgb_img.shape[:2]
        target_size = 256
        new_width = (original_width // target_size) * target_size
        new_height = (original_height // target_size) * target_size
        new_width = max(target_size, new_width)
        new_height = max(target_size, new_height)
        
        if new_width != original_width or new_height != original_height:
            rgb_img = cv2.resize(rgb_img, (new_width, new_height))
        
        lab_img = color.rgb2lab(rgb_img)
        l_channel = lab_img[:, :, 0]
        l_normalized = (l_channel / 50.0) - 1.0
        l_tensor = torch.from_numpy(l_normalized).float().unsqueeze(0).unsqueeze(0)
        dummy_ab = torch.zeros((1, 2, l_tensor.shape[2], l_tensor.shape[3]), dtype=torch.float)
        data = {'A': l_tensor, 'B': dummy_ab, 'A_paths': 'uploaded_image', 'B_paths': 'dummy_path'}
        self.model.set_input(data)
        self.model.test()
        
        fake_ab = self.model.fake_B.detach().cpu().numpy()[0]
        fake_ab = fake_ab * 110.0
        l_channel = l_channel.reshape(l_channel.shape[0], l_channel.shape[1], 1)
        fake_ab = np.transpose(fake_ab, (1, 2, 0))
        
        if fake_ab.shape[0] != l_channel.shape[0] or fake_ab.shape[1] != l_channel.shape[1]:
            fake_ab = cv2.resize(fake_ab, (l_channel.shape[1], l_channel.shape[0]))
        
        colorized_lab = np.concatenate([l_channel, fake_ab], axis=2)
        colorized_rgb = color.lab2rgb(colorized_lab) * 255
        colorized_rgb = colorized_rgb.astype(np.uint8)
        
        if new_width != original_width or new_height != original_height:
            colorized_rgb = cv2.resize(colorized_rgb, (original_width, original_height))
        
        return colorized_rgb

# Initialize the model
def load_model():
    model_path = './checkpoints'
    model_name = 'tongyong_l2ab_4'
    model_epoch = '35'
    use_cuda = torch.cuda.is_available()
    return ColorizeModel(model_path, model_name, model_epoch, use_cuda)

model = load_model()

# Create a directory for temporary files if it doesn't exist
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_files")
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/colorize")
async def colorize_image(file: UploadFile = File(...)):
    """API endpoint to colorize an uploaded image"""
    # Create unique filenames for input and output
    input_filename = f"{next(tempfile._get_candidate_names())}.png"
    output_filename = f"{next(tempfile._get_candidate_names())}.png"
    
    input_path = os.path.join(TEMP_DIR, input_filename)
    output_path = os.path.join(TEMP_DIR, output_filename)
    
    try:
        # Read the uploaded file and save it
        contents = await file.read()
        with open(input_path, 'wb') as f:
            f.write(contents)
        
        # Process the image
        input_image = Image.open(input_path)
        colorized_rgb = model.process_image(input_image)
        
        # Save the result
        cv2.imwrite(output_path, cv2.cvtColor(colorized_rgb, cv2.COLOR_RGB2BGR))
        
        # Create a response
        return FileResponse(
            path=output_path,
            filename=output_filename,
            media_type="image/png",
            background=None  # Don't run in background to ensure file exists when response is sent
        )
        
    except Exception as e:
        # Clean up files in case of error
        for path in [input_path, output_path]:
            if os.path.exists(path):
                os.unlink(path)
        raise HTTPException(status_code=500, detail=str(e))

# Cleanup function to delete old temporary files
@app.on_event("startup")
async def cleanup_old_files():
    """Clean up old temporary files at startup"""
    try:
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    except Exception as e:
        print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9700) 