"""Create a hyperspectral camera with DeepLens.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from authors).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.
"""

from deeplens import HSICamera

hsi_cam = HSICamera(
    lens_file="./lenses/paraxiallens/doelens_hsi.json",
    sensor_file="./sensors/flir/BFS-U3-200S7C-C.json",
)
hsi_cam.vis_doe()
hsi_cam.vis_psf(wvln_spectral=[0.4, 0.5, 0.6, 0.7], depth=-1000)
