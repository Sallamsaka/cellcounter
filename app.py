import numpy as np
import torch
import gradio as gr
from PIL import Image
import torch.nn as nn
import os

# Defining the U-Net model

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.reduce = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False)
        up_ch = in_ch // 2

        self.conv = DoubleConv(up_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.reduce(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNetDensity(nn.Module):
    def __init__(self, in_channels=1, base=16, out_activation="relu"):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base*2, base*4, base*8, base*16

        self.inc = DoubleConv(in_channels, c1)
        self.d1  = Down(c1, c2)
        self.d2  = Down(c2, c3)
        self.d3  = Down(c3, c4)
        self.d4  = Down(c4, c5)

        self.u1 = Up(c5, c4, c4)
        self.u2 = Up(c4, c3, c3)
        self.u3 = Up(c3, c2, c2)
        self.u4 = Up(c2, c1, c1)

        self.outc = nn.Conv2d(c1, 1, kernel_size=1)

        if out_activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif out_activation == "softplus":
            self.act = nn.Softplus()
        else:
            print("WRONG THING")
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)

        x = self.u1(x5, x4)
        x = self.u2(x,  x3)
        x = self.u3(x,  x2)
        x = self.u4(x,  x1)

        x = self.outc(x)
        return self.act(x)

def pos1d(L, P, stride):
        if L <= P:
            return [0]
        pos = list(range(0, L - P + 1, stride))
        if pos[-1] != L - P:
            pos.append(L - P)
        return pos

def predict_image_density_and_count(model, x_full, P=256, stride=256):

    H, W = x_full.shape

    tops = pos1d(H, P, stride)
    lefts = pos1d(W, P, stride)

    patch_indices = [(t, l) for t in tops for l in lefts]  
    
    acc = torch.zeros((H, W), dtype=torch.float32)
    wgt = torch.zeros((H, W), dtype=torch.float32)

    for top, left in patch_indices:
        x = x_full[top:top+P, left:left+P].astype(np.float32, copy=False)
        h_valid, w_valid = x.shape

        if h_valid < P or w_valid < P:
            x_pad = np.zeros((P, P), dtype=np.float32)
            x_pad[:h_valid, :w_valid] = x
            x = x_pad

        xt = torch.from_numpy(x)[None, None, :, :]

        pred = model(xt)[0, 0]

        acc[top:top+h_valid, left:left+w_valid] += pred[:h_valid, :w_valid]
        wgt[top:top+h_valid, left:left+w_valid] += 1.0

    full_density = acc / wgt.clamp_min(1.0)
    full_count = full_density.sum().item()

    return full_density.numpy(), full_count

def build_model():
    model = UNetDensity(in_channels=1, base=16, out_activation="relu")
    return model

best_model = None

def get_best_model(weights_path="best.pt"):
    global best_model

    if best_model is not None:
        return best_model

    ckpt = torch.load(weights_path, map_location="cpu")

    model = build_model()
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    best_model = model
    return model

def display_scaling(density: np.ndarray) -> Image.Image:

    mx = float(density.max())
    if mx > 0:
        density = density / mx

    return Image.fromarray((255 * density).astype(np.uint8), mode="L")

DEMO_DIR = "demo_images"

def load_demo_paths():
    paths = []
    for image in sorted(os.listdir(DEMO_DIR)):
        paths.append(os.path.join(DEMO_DIR, image))
    return paths

demo_paths = load_demo_paths()

def gradio_predict(img_pil):
    if img_pil is None:
        return None, None, "Select an image from the gallery or upload your own."
    
    model = get_best_model()

    x_full = np.array(img_pil.convert("L"), dtype=np.float32) / 255.0

    density, count = predict_image_density_and_count(model, x_full)

    density_img = display_scaling(density)
    note = "Contrast of density map is scaled for better visualization."

    return density_img, count, note

@torch.inference_mode()
def on_gallery_select(evt: gr.SelectData):
    path = demo_paths[evt.index]
    img = Image.open(path)
    return gradio_predict(img)

@torch.inference_mode()
def on_upload_change(img_pil):
    return gradio_predict(img_pil)

with gr.Blocks() as demo:
    gr.Markdown(
        "# Cell Density Estimator\n"
        "Click a gallery image or upload your own image to get the predicted density map and count.\n"
    )

    with gr.Row():
        with gr.Column():
            gallery = gr.Gallery(
                value=demo_paths,
                label="Example images (click to select)",
                columns=4,
                height=320,
            )

            upload = gr.Image(type="pil", label="Or upload your own image")


        with gr.Column():
            out_density = gr.Image(type="pil", label="Predicted density map")
            out_count = gr.Number(label="Predicted count")
            out_note = gr.Markdown("")

    gallery.select(
        on_gallery_select,
        inputs=None,
        outputs=[out_density, out_count, out_note],
    )

    upload.change(
        on_upload_change,
        inputs=[upload],
        outputs=[out_density, out_count, out_note],
    )

if __name__ == "__main__":
    demo.launch()