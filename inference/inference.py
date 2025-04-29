import dash
from dash import dcc, html, Input, Output
import base64
import io
from PIL import Image
import torch
from torchvision import transforms
from omegaconf import OmegaConf

from inference_utils import load_model, run_inference, visualize_result

# Load configuration and model.
cfg = OmegaConf.load("configs/train.yaml")
device = torch.device(cfg.hardware.device if torch.cuda.is_available() else "cpu")
model = load_model(cfg, device)
model.eval()

# Image preprocessing (preserving aspect ratio)
transform = transforms.Compose([
    transforms.Resize(max(cfg.dataset.transform_size), interpolation=Image.BILINEAR),
    transforms.ToTensor()
])

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Segmentation Dashboard"
app.css.config.serve_locally = False

# Dark theme layout
app.layout = html.Div(
    style={
        'backgroundColor': '#121212',
        'color': '#e0e0e0',
        'fontFamily': 'Segoe UI, sans-serif',
        'minHeight': '100vh',
        'paddingBottom': '50px'
    },
    children=[
        html.H1("Semantic Segmentation Dashboard", style={
            'textAlign': 'center',
            'paddingTop': '20px',
            'color': '#f5f5f5',
            'textShadow': '1px 1px 3px #000'
        }),

        dcc.Upload(
            id='upload-image',
            children=html.Div([
                "📤 Drag and Drop or ",
                html.A("Select an Image", style={'color': '#f48fb1'})
            ]),
            style={
                'width': '60%',
                'height': '120px',
                'lineHeight': '120px',
                'borderWidth': '2px',
                'borderStyle': 'dashed',
                'borderRadius': '10px',
                'borderColor': '#555',
                'textAlign': 'center',
                'margin': '30px auto',
                'backgroundColor': '#1e1e1e',
                'color': '#e0e0e0',
                'fontSize': '18px'
            },
            multiple=False
        ),

        html.Div(id='output-image'),

        html.Div([
            dcc.Graph(
                id='predicted-mask',
                style={
                    'display': 'none',
                    'backgroundColor': '#121212'
                },
                config={'displayModeBar': False}
            )
        ], style={
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
            'marginTop': '20px'
        })
    ]
)


def parse_image(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return Image.open(io.BytesIO(decoded)).convert("RGB")


@app.callback(
    Output('predicted-mask', 'figure'),
    Output('predicted-mask', 'style'),
    Input('upload-image', 'contents')
)
def update_output(content):
    if content is None:
        return {"data": [], "layout": {}}, {'display': 'none'}

    image = parse_image(content)
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_mask = run_inference(model, image_tensor, cfg.model.inference_prebuilt)

    fig = visualize_result(image, pred_mask)
    return fig, {
        'display': 'block',
        'margin': '0 auto',
        'backgroundColor': '#121212'
    }


if __name__ == '__main__':
    try:
        app.run(debug=True)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Exiting gracefully.")
        exit(0)
