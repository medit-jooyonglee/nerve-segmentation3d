import os, sys
import numpy as np

def handle_base_path():
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


handle_base_path()

import vtk_utils


def test_save_image():
    x = np.random.randn(50, 50, 50)
    print(x.shape)
    vtk_utils.show_actors([
        x,
    ], show=False, image_save=True, savename='temp.png')

