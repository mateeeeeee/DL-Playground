import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from typing import Optional

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

TENSOR_TO_PIL = transforms.ToPILImage()

def gui_image_loader(image_path: str, process_size: int) -> Optional[torch.Tensor]:
    try:
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return None

        image = Image.open(image_path).convert('RGB')

        loader = transforms.Compose([
            transforms.Resize((process_size, process_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor()])

        image_tensor = loader(image).unsqueeze(0)

        return image_tensor.to(device, torch.float)

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading or processing image {os.path.basename(image_path)}: {e}")
        return None

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    image = tensor.cpu().clone().detach()
    image = image.squeeze(0)
    image = torch.clamp(image, 0, 1)
    pil_image = TENSOR_TO_PIL(image)
    return pil_image