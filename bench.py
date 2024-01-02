import argparse
from pathlib import Path

import numpy as np
import realesrgan
import spandrel
import torch
from PIL import Image, ImageChops, ImageOps
from basicsr.archs.rrdbnet_arch import RRDBNet


def pil_image_to_torch_bgr(img: Image.Image) -> torch.Tensor:
    img = np.array(img.convert("RGB"))
    img = img[:, :, ::-1]  # flip RGB to BGR
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img) / 255  # Rescale to [0, 1]
    return torch.from_numpy(img)


def torch_bgr_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4:
        # If we're given a tensor with a batch dimension, squeeze it out
        # (but only if it's a batch of size 1).
        if tensor.shape[0] != 1:
            raise ValueError(f"{tensor.shape} does not describe a BCHW tensor")
        tensor = tensor.squeeze(0)
    assert tensor.ndim == 3, f"{tensor.shape} does not describe a CHW tensor"
    # TODO: is `tensor.float().cpu()...numpy()` the most efficient idiom?
    arr = tensor.float().cpu().clamp_(0, 1).numpy()  # clamp
    arr = 255.0 * np.moveaxis(arr, 0, 2)  # CHW to HWC, rescale
    arr = arr.round().astype(np.uint8)
    arr = arr[:, :, ::-1]  # flip BGR to RGB
    return Image.fromarray(arr, "RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--device", "-d", type=str, required=True)
    ap.add_argument("--out-dir", "-o", type=str, default="out")
    args = ap.parse_args()
    device = torch.device(args.device)
    path = "./RealESRGAN_x4plus_anime_6B.pth"
    s_model = spandrel.ModelLoader(device=device).load_from_file(path)
    re_model = realesrgan.RealESRGANer(
        scale=4,
        model_path=path,
        device=device,
        model=RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
        ),
        pre_pad=0,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Opening...")
    source_img = Image.open(args.image).convert("RGB")
    print(source_img)
    with torch.no_grad():
        print("realesrgan...")
        img_re = Image.fromarray(re_model.enhance(np.array(source_img))[0])
        img_re.save(out_dir / f"out_realesrgan_{device}.png")
        print("=>", img_re)

        print("spandrel...")
        source_img_t = (
            pil_image_to_torch_bgr(source_img).float().unsqueeze(0).to(device)
        )
        img_sp = torch_bgr_to_pil_image(s_model(source_img_t))
        img_sp.save(out_dir / f"out_spandrel_{device}.png")
        print("=>", img_sp)

        print("diffs...")
        img_abs_diff = ImageChops.difference(img_re, img_sp)
        img_abs_diff.save(out_dir / f"out_abs_diff_{device}.png")
        img_abs_diff_norm = ImageOps.equalize(img_abs_diff)
        img_abs_diff_norm.save(out_dir / f"out_abs_diff_norm_{device}.png")
    print("done.")


if __name__ == "__main__":
    main()
