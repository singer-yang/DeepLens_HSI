import json
import torch
from deeplens.diffraclens import DiffractiveLens
from deeplens.optics.render_psf import render_psf
from deeplens.camera import Camera, Renderer
from deeplens.sensor import RGBSensor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.io as sio
import numpy as np


class HSICamera(Renderer):
    def __init__(self, lens_file, sensor_file, device=None):
        super().__init__(device=device)

        # RGB sensor
        sensor_config = json.load(open(sensor_file))
        sensor = RGBSensor.from_config(sensor_config)
        self.sensor = sensor.to(self.device)

        # Single DOE lens
        self.lens = DiffractiveLens(
            filename=lens_file,
            sensor_res=sensor.res,
            sensor_size=sensor.size,
            device=self.device,
        )
        self.lens.surfaces[0].phase_map = self.read_mat()

    def read_mat(self, height_map_path="./lenses/paraxiallens/planar_doe.mat"):
        """Read phase map from a .mat file. This function only works for the BUPT DOE."""
        # Load the .mat file
        mat_data = sio.loadmat(height_map_path)
        data_keys = [k for k in mat_data.keys() if not k.startswith("__")]
        doe_data = None
        for key in data_keys:
            if isinstance(mat_data[key], np.ndarray) and mat_data[key].size > 1:
                doe_data = mat_data[key]
                break
        height_map = doe_data[0, :, :, 0]  # shape (1024, 1024), data range [0, 2000]

        # Calculate relative height map ([um]) for non-zero values
        non_zero_mask = height_map > 0
        if np.any(non_zero_mask):
            min_non_zero = np.min(height_map[non_zero_mask])
            height_map[non_zero_mask] -= min_non_zero

        # Convert height map to phase map
        doe = self.lens.surfaces[0]
        phase_map0 = height_map * (doe.n0 - 1) * 2 * torch.pi / doe.wvln0
        phase_map0 = torch.from_numpy(phase_map0).to(self.device)

        return phase_map0

    def vis_psf(self, wvln_spectral, psf_ks=201, depth=float("inf")):
        """Visualize PSF for each spectral channel with colors corresponding to wavelengths.
        Wavelengths from ~400nm (blue/purple) to ~700nm (red).
        """
        # Calculate PSF for each spectral channel
        psf_spectral = torch.zeros((len(wvln_spectral), psf_ks, psf_ks)).to(self.device)
        for i, wvln in enumerate(wvln_spectral):
            psf_chan = self.lens.psf(depth=depth, wvln=wvln, ks=psf_ks, upsample_factor=2)
            psf_spectral[i, :, :] = psf_chan

        # Plot PSFs
        norm = mcolors.Normalize(vmin=0.4, vmax=0.7)
        cmap_name = "jet"
        fig = plt.figure(figsize=(15, 5))
        for i, wvln in enumerate(wvln_spectral):
            plt.subplot(1, len(wvln_spectral), i + 1)
            color = plt.cm.get_cmap(cmap_name)(norm(wvln))
            colors = [(0, 0, 0, 1), color]
            custom_cmap = mcolors.LinearSegmentedColormap.from_list(
                f"custom_{i}", colors
            )
            plt.imshow(psf_spectral[i, :, :].cpu().numpy(), cmap=custom_cmap)
            wvln_nm = wvln * 1000 if wvln < 10 else wvln
            plt.title(f"PSF at {wvln_nm:.0f} nm")
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        fig.savefig(f"./psf_spectral_{depth}um.png", dpi=300, bbox_inches="tight")

        plt.close("all")

    def spectral2rgb(self, img_spectral):
        """Convert reconstrcuted spectral image to rgb image for visualization."""
        img_rgb_raw = self.sensor.response_curve(img_spectral)
        return img_rgb_raw

    def render_lens(self, img_spectral, wvln_spectral, depth=float("inf"), psf_ks=201):
        """Render lens blur for each spectral channel.

        Args:
            img_spectral (torch.Tensor): Input image. Shape: (B, C, H, W)
            wvln_spectral (torch.Tensor): Wavelength. Shape: (C,)
            psf_ks (int, optional): PSF kernel size. Defaults to 51.

        Returns:
            img_render (torch.Tensor): Rendered image. Shape: (B, C, H, W)

        Note:
            [1] When creating a diffractive lens, we change the default dtype to double. Here we need to convert it back.
        """
        # Calculate PSF for all spectral channels
        num_spectral = len(wvln_spectral)
        psf_spectral = (
            torch.zeros((num_spectral, psf_ks, psf_ks))
            .to(img_spectral.device)
            .to(img_spectral.dtype)
        )
        for i, wvln in enumerate(wvln_spectral):
            psf_chan = self.lens.psf(
                depth=depth, wvln=wvln, ks=psf_ks, upsample_factor=2
            )
            psf_spectral[i, :, :] = psf_chan

        # PSF convolution by each spectral channel
        img_render = render_psf(img_spectral, psf_spectral)
        return img_render

    def render(self, data_dict):
        """Render a blurry and noisy RGB image batch from spectral data.

        Args:
            data_dict (dict): Dictionary containing essential information for image simulation, for example:
                {
                    "wvln": wvln (torch.Tensor), (B, C)
                    "img": rgb image (torch.Tensor), (B, C, H, W), [0, 1]
                }

        Returns:
            img_spectral (torch.Tensor): Rendered spectral image. Shape: (B, C, H, W)
            img_rgb_raw (torch.Tensor): Rendered RGB image. Shape: (B, 3, H, W)
        """
        data_dict = self.move_to_device(data_dict)
        wvln_spectral = data_dict["wvln"][0].tolist()  # (C,)
        img_spectral = data_dict["img"]  # (B, C, H, W)

        # Render lens blur for all spectral channels
        img_spectral_render = self.render_lens(img_spectral, wvln_spectral)

        # Simulate sensor response (raw space)
        img_rgb_raw = self.sensor.response_curve(img_spectral_render)

        # Simulate sensor noise
        # img_rgb_noise = sensor.simu_noise(img_rgb_raw, iso)

        return img_rgb_raw, img_spectral
