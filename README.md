* create a virtualenv
* install deps (Torch in your preferred way, and then `pip install -r requirements.in`)
* get `RealESRGAN_x4plus_anime_6B.pth` from e.g. [here](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth) and put it in the project directory
* get a test image that fits in your Torch device (this bench doesn't do tiling)
* run `python bench.py --device DEVICE image.png` (device is e.g. `cpu`, `cuda`, `mps`)
* Use your favored image (diff) tool to look at the `out/*` images
