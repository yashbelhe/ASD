import os
import numpy as np
import skimage.io
import subprocess
import tempfile
import torch

def imwrite(img, filename, gamma = 2.2, normalize = False):
    """
    Save an image to a file with the origin at the lower left corner.
    Taken from https://github.com/BachiLi/diffvg/blob/85802a71fbcc72d79cb75716eb4da4392fd09532/pydiffvg/image.py
    """
    directory = os.path.dirname(filename)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)

    # set origin to lower left corner
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.dim() == 3:
            img = img.flip(0)
        elif img.dim() == 4:
            img = img.flip(1)
    else:
        img = np.flipud(img)

    if not isinstance(img, np.ndarray):
        img = img.data.numpy()
    if normalize:
        img_rng = np.max(img) - np.min(img)
        if img_rng > 0:
            img = (img - np.min(img)) / img_rng
    img = np.clip(img, 0.0, 1.0)
    if img.ndim==2:
        img=np.expand_dims(img,2)
    img[:, :, :3] = np.power(img[:, :, :3], 1.0/gamma)
    skimage.io.imsave(filename, (img * 255).astype(np.uint8))

def vidwrite(imgs, filename, fps=1):
    # Create a temporary directory to store frames
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save each frame as a PNG file
        for i, img in enumerate(imgs):
            frame_path = os.path.join(tmpdir, f"frame_{i:05d}.png")
            imwrite(img, frame_path)
        
        # Use ffmpeg to create a GIF from the frames
        cmd = [
            "ffmpeg", 
            "-y",  # Overwrite output file if it exists
            "-f", "image2",  # Force input format to image2
            "-framerate", str(fps),  # Set the frame rate
            "-i", os.path.join(tmpdir, "frame_%05d.png"),  # Input pattern
            filename  # Output file
        ]
        
        subprocess.run(cmd, check=True)
    
