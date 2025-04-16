import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import queue
import time
from typing import List, Tuple
from image_utils import gui_image_loader, tensor_to_pil

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA (GPU) is available and will be used")
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU")

class StyleTransferError(Exception):
    """Custom exception for style transfer specific errors"""
    pass

class Normalization(nn.Module):
    """
    Normalizes an image tensor using mean and standard deviation.
    Required step before feeding images into models pre-trained on ImageNet.
    """
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1).to(device)
        self.std = std.clone().detach().view(-1, 1, 1).to(device)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Applies normalization: (img - mean) / std."""
        return (img - self.mean) / self.std

class ContentLoss(nn.Module):
    """
    Computes the content loss between the feature map of the input image
    and the feature map of the original content image
    """
    def __init__(self, target_feature: torch.Tensor):
        super().__init__()
        self.target = target_feature.detach()
        self.loss: torch.Tensor = torch.tensor(0.0, device=device)

    def forward(self, current_feature: torch.Tensor) -> torch.Tensor:
        self.loss = F.mse_loss(current_feature, self.target)
        return current_feature

class StyleLoss(nn.Module):
    """
    Computes the style loss between the Gram matrix of the input image's
    feature map and the Gram matrix of the style image's feature map
    """
    def __init__(self, target_feature: torch.Tensor):
        super().__init__()
        self.target_gram = self._gram_matrix(target_feature).detach()
        self.loss: torch.Tensor = torch.tensor(0.0, device=device)

    @staticmethod
    def _gram_matrix(input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = input_tensor.size()
        features = input_tensor.view(batch_size * num_channels, height * width)
        gram = torch.mm(features, features.t())
        return gram.div(batch_size * num_channels * height * width)

    def forward(self, current_feature: torch.Tensor) -> torch.Tensor:
        current_gram = self._gram_matrix(current_feature)
        self.loss = F.mse_loss(current_gram, self.target_gram)
        return current_feature

class StyleTransfer:
    """
    Encapsulates the neural style transfer algorithm using a pre-trained VGG network
    """
    CONTENT_LAYERS_DEFAULT: List[str] = ['conv_4']
    STYLE_LAYERS_DEFAULT: List[str] = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    CNN_NORMALIZATION_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device)
    CNN_NORMALIZATION_STD = torch.tensor([0.229, 0.224, 0.225], device=device)

    def __init__(self):
        """Loads the pre-trained VGG19 model features."""
        try:

            weights = models.VGG19_Weights.IMAGENET1K_V1
            cnn = models.vgg19(weights=weights).features

            self.cnn = cnn.to(device).eval()
            print(f"VGG19 model loaded successfully onto {device}.")
        except Exception as e:
            print(f"FATAL: Failed to load VGG19 model: {e}")

            raise StyleTransferError(f"Failed to load VGG19 model: {e}") from e

    def _get_style_model_and_losses(self,
                                    style_img: torch.Tensor,
                                    content_img: torch.Tensor,
                                    content_layers: List[str] = CONTENT_LAYERS_DEFAULT,
                                    style_layers: List[str] = STYLE_LAYERS_DEFAULT
                                    ) -> Tuple[nn.Module, List[StyleLoss], List[ContentLoss]]:
        """
        Builds a sequential model by adding normalization and custom loss layers
        to a copy of the VGG feature extractor.
        """
        cnn_copy = copy.deepcopy(self.cnn)

        normalization = Normalization(self.CNN_NORMALIZATION_MEAN, self.CNN_NORMALIZATION_STD)
        model = nn.Sequential(normalization)

        content_losses: List[ContentLoss] = []
        style_losses: List[StyleLoss] = []

        conv_counter = 0
        relu_counter = 0
        pool_counter = 0
        bn_counter = 0

        for layer in cnn_copy.children():
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
                name = f'conv_{conv_counter}'
            elif isinstance(layer, nn.ReLU):
                relu_counter += 1
                name = f'relu_{relu_counter}'

                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                pool_counter += 1
                name = f'pool_{pool_counter}'
            elif isinstance(layer, nn.BatchNorm2d):
                 bn_counter += 1
                 name = f'bn_{bn_counter}'
            else:

                layer_type = layer.__class__.__name__
                print(f"Warning: Unrecognized layer type encountered: {layer_type}")
                name = f'unknown_{layer_type}'

            model.add_module(name, layer)

            if name in content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module(f"content_loss_{conv_counter}", content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module(f"style_loss_{conv_counter}", style_loss)
                style_losses.append(style_loss)

            if name == max(content_layers + style_layers, key=lambda n: int(n.split('_')[-1])):
                 break

        for param in model.parameters():
            param.requires_grad_(False)

        return model, style_losses, content_losses

    def _get_input_optimizer(self, input_img: torch.Tensor) -> optim.Optimizer:
        """
        Creates the LBFGS optimizer commonly used for style transfer,
        which optimizes the input image tensor directly.
        """

        optimizer = optim.LBFGS([input_img.requires_grad_(True)], lr=1)
        return optimizer


    def run_style_transfer_task(self,
                                content_path: str,
                                style_path: str,
                                process_size: int,
                                status_queue: queue.Queue,
                                result_queue: queue.Queue,
                                num_steps: int = 300,
                                style_weight: float = 1e6,
                                content_weight: float = 1.0
                                ) -> None:
        #todo add desciption
        try:
            status_queue.put("Loading images...")
            content_img_tensor = gui_image_loader(content_path, process_size)
            style_img_tensor = gui_image_loader(style_path, process_size)

            if content_img_tensor is None or style_img_tensor is None:
                 raise StyleTransferError("Failed to load one or both images")


            if content_img_tensor.size() != style_img_tensor.size():
                msg = (f"Content tensor size {content_img_tensor.size()} "
                       f"does not match style tensor size {style_img_tensor.size()} "
                       f"after loading/resizing")
                print(f"Warning: {msg}")

            input_img = content_img_tensor.clone()

            status_queue.put("Building style transfer model...")
            model, style_losses, content_losses = self._get_style_model_and_losses(
                style_img_tensor, content_img_tensor
            )

            input_img.requires_grad_(True)

            model.requires_grad_(False)

            optimizer = self._get_input_optimizer(input_img)

            status_queue.put(f"Optimizing for {num_steps} steps...")
            print(f"Starting optimization: {num_steps=}, {style_weight=}, {content_weight=}")

            run = [0]
            start_time = time.time()

            while run[0] < num_steps:

                def closure() -> float:
                    with torch.no_grad():
                        input_img.clamp_(0, 1)

                    optimizer.zero_grad()
                    model(input_img)

                    s_loss: torch.Tensor = torch.tensor(0.0, device=device)
                    c_loss: torch.Tensor = torch.tensor(0.0, device=device)

                    for sl in style_losses:
                        s_loss += sl.loss

                    for cl in content_losses:
                        c_loss += cl.loss


                    s_loss *= style_weight
                    c_loss *= content_weight

                    total_loss = s_loss + c_loss
                    total_loss.backward()

                    run[0] += 1

                    if run[0] % 50 == 0 or run[0] == 1:
                        elapsed = time.time() - start_time
                        status_msg = (f"Step {run[0]}/{num_steps} - "
                                      f"Loss: {total_loss.item():.3f} "
                                      f"(S: {s_loss.item():.3f}, C: {c_loss.item():.3f}) "
                                      f"({elapsed:.1f}s)")
                        status_queue.put(status_msg)
                        print(status_msg)

                    return total_loss.item()

                optimizer.step(closure)


            with torch.no_grad():
                input_img.clamp_(0, 1)

            status_queue.put("Optimization finished.")
            print("Optimization finished.")

            final_pil_image = tensor_to_pil(input_img)
            result_queue.put(final_pil_image)

        except StyleTransferError as e:

            error_msg = f"Style Transfer Error: {e}"
            print(error_msg)
            status_queue.put(f"Error: {e}")
            result_queue.put(e)
        except Exception as e:

            import traceback
            error_msg = f"Unexpected Error: {e}\n{traceback.format_exc()}"
            print(error_msg)
            status_queue.put(f"Unexpected Error: {e}")
            result_queue.put(e)