import torch
import time
from PIL import Image
import torchvision.transforms.functional as TF

import config
from models.srcnn import SRCNN

def preprocess_image_for_inference(image_pil, device):
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    img_ycbcr = image_pil.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()
    input_y = TF.to_tensor(img_y).unsqueeze(0).to(device)
    return input_y, img_cb, img_cr, image_pil.size

def postprocess_output(output_y_tensor, img_cb, img_cr):
    output_y_tensor = output_y_tensor.squeeze(0).cpu().detach()
    output_y_tensor = torch.clamp(output_y_tensor, 0.0, 1.0)
    output_y_img = TF.to_pil_image(output_y_tensor, mode='L')

    target_size_wh = output_y_img.size
    img_cb_resized = img_cb.resize(target_size_wh, Image.Resampling.BICUBIC)
    img_cr_resized = img_cr.resize(target_size_wh, Image.Resampling.BICUBIC)

    final_img = Image.merge('YCbCr', (output_y_img, img_cb_resized, img_cr_resized)).convert('RGB')
    return final_img

def upscale_image(input_image_pil: Image.Image, model: SRCNN, scale_factor: int, device: torch.device):
    """ Performs SRCNN upscale on the input image; Assumes the model is trained for the given scale_factor """
    if not model:
        raise ValueError("SRCNN model not provided for upscaling.")

    start_time = time.time()

    target_w = input_image_pil.width * scale_factor
    target_h = input_image_pil.height * scale_factor

    bicubic_img = input_image_pil.resize((target_w, target_h), Image.Resampling.BICUBIC)
    input_y, img_cb, img_cr, _ = preprocess_image_for_inference(bicubic_img, device)

    model.eval() 
    with torch.no_grad():
        output_y = model(input_y)
    final_img = postprocess_output(output_y, img_cb, img_cr)

    end_time = time.time()
    print(f"SRCNN Upscaling x{scale_factor} took {end_time - start_time:.2f}s")

    return final_img

def upscale_bicubic(input_image_pil: Image.Image, scale_factor: int):
    start_time = time.time()
    target_w = input_image_pil.width * scale_factor
    target_h = input_image_pil.height * scale_factor
    bicubic_img = input_image_pil.resize((target_w, target_h), Image.Resampling.BICUBIC)
    end_time = time.time()
    print(f"Bicubic Upscaling x{scale_factor} took {end_time - start_time:.2f}s")
    return bicubic_img
