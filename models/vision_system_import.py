import torch
import torch.nn as nn
import warnings
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor
from transformers.utils import logging as hf_logging
from torchvision.transforms.functional import to_pil_image

# ─── Suppress Warnings ──────────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()

class VisionSystemImport(nn.Module):
    def __init__(self, 
                 model_name="facebook/mask2former-swin-large-ade-semantic",
                 image_size=(512, 512),
                 use_fp16=True,
                 device="cuda"):
        """
        Constructs a segmentation VisionSystem using Hugging Face's semantic Mask2Former.

        Args:
            model_name (str): Semantic Mask2Former model (e.g., ADE20K).
            image_size (tuple): Resize all inputs to this size to reduce memory usage.
            use_fp16 (bool): Enable FP16 inference (only effective on CUDA).
            device (str): Device to run on.
        """
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.use_fp16 = use_fp16

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(self.device)

        if self.use_fp16 and self.device == "cuda":
            self.model.half()

    def forward(self, images, masks=None):
        """
        Args:
            images (torch.Tensor): Tensor of shape [B, C, H, W].
            masks (torch.Tensor): Tensor of shape [B, H, W] with class indices (if training).

        Returns:
            - If training: scalar loss tensor.
            - If eval: List of per-image segmentation maps [H, W].
        """
        # Convert tensor to PIL images
        if isinstance(images, torch.Tensor):
            if images.device.type != "cpu":
                images = images.cpu()
            images = images.float()
            images = [to_pil_image(img) for img in images]

        # Resize images
        images = [img.resize(self.image_size) for img in images]

        # Prepare processor input
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # If training, add masks
        if self.training:
            if masks is None:
                raise ValueError("Masks must be provided in training mode.")

            if masks.device.type != "cpu":
                masks = masks.cpu()
            masks = masks.long()

            # Resize masks to match image_size
            masks_resized = torch.nn.functional.interpolate(
                masks.unsqueeze(1).float(), size=self.image_size[::-1], mode="nearest"
            ).squeeze(1).long()

            inputs["labels"] = masks_resized.to(self.device)

        # FP16 inference
        if self.use_fp16 and self.device == "cuda":
            inputs = {k: v.half() if torch.is_tensor(v) else v for k, v in inputs.items()}

        outputs = self.model(**inputs)

        if self.training:
            return outputs.loss
        else:
            # Eval mode → post-process segmentation
            target_sizes = [img.size[::-1] for img in images]
            return self.processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)

# # ─── Quick Test ─────────────────────────────────────────────
# if __name__ == "__main__":
#     from PIL import Image
#     import requests
#     import matplotlib.pyplot as plt

#     # Sample image
#     url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#     image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

#     model = VisionSystemImport(
#         model_name="facebook/mask2former-swin-large-ade-semantic",
#         image_size=(512, 512),
#         use_fp16=True,
#         device="cuda" if torch.cuda.is_available() else "cpu"
#     )

#     # 🔍 Eval Test
#     model.eval()
#     result = model([image])
#     print("Eval mask shape:", result[0].shape)
#     plt.imshow(result[0], cmap="jet")
#     plt.axis("off")
#     plt.title("Semantic Segmentation Output")
#     plt.show()

#     # 🧪 Training Test
#     model.train()
#     dummy_image = torch.randn(2, 3, 256, 256)
#     dummy_mask = torch.randint(0, 151, (2, 256, 256))  # ADE20K has 150+1 classes
#     loss = model(dummy_image, dummy_mask)
#     print("Training loss:", loss.item())
