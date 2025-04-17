

import torch
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

from utils.visualizer import show_comparison


def load_model(cfg, device):
    """
    Loads model based on cfg.model.inference_prebuilt flag.
    """
    if cfg.model.inference_prebuilt:
        from torchvision.models.segmentation import fcn_resnet50
        from torchvision.models import resnet50
        print("[INFO] Using torchvision's FCN-ResNet50 model for inference.")
        backbone = resnet50(weights=None)
        model = fcn_resnet50(weights=None, weights_backbone=None, num_classes=cfg.model.num_classes).to(device)
    else:
        from models.segmentation import Segmentation
        print("[INFO] Using custom Segmentation model (Swin + Mask2Former).")
        model = Segmentation(config_path=cfg.paths.model_config)

    model.load_state_dict(torch.load(cfg.paths.output_inference_model, map_location=device))
    return model.to(device)


def run_inference(model, image_tensor, inbuilt=False):
    """
    Performs forward pass and returns predicted mask.
    """
    outputs = model(image_tensor)
    if inbuilt:
        pred_mask = torch.argmax(outputs["out"], dim=1).squeeze().cpu().numpy()
    else:
        pred_mask = torch.argmax(outputs["pred_masks"].squeeze(0), dim=0).cpu().numpy()
    return pred_mask


def visualize_result(image_pil, pred_mask, cfg=None):
    """
    Creates a 2x2 grid figure using show_comparison and converts it to a Plotly-compatible format.
    """

    # Resize the image to match prediction shape
    original = np.array(image_pil.resize((pred_mask.shape[1], pred_mask.shape[0])))
    dummy_gt = np.zeros_like(pred_mask)

    class_info_path = cfg.paths.weights_path if cfg is not None else None

    fig = show_comparison(original, dummy_gt, pred_mask, class_info=class_info_path)

    # Convert to Plotly-compatible base64 image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return {
        'data': [],
        'layout': {
            'images': [{
                'source': f'data:image/png;base64,{encoded_image}',
                'xref': 'paper',
                'yref': 'paper',
                'x': 0, 'y': 1,
                'sizex': 1,
                'sizey': 1,
                'xanchor': 'left',
                'yanchor': 'top',
                'layer': 'below',
                'sizing': 'contain'
            }],
            'xaxis': {'visible': False},
            'yaxis': {'visible': False},
            'margin': dict(l=0, r=0, t=0, b=0),
            'height': 768,
            'width': 768,
            'autosize': False
        }
    }
