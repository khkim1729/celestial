"""
HAEDAL_Slicer.py
----------------
3D MRI volume → 2D slices per axis.

For each axis (axial / coronal / sagittal):
  1. Find slice index n with the largest tumor area in the segmentation mask.
  2. Extract that slice from all 4 modalities.
  3. Tile the 4 slices into a 2×2 montage → resize to img_size × img_size.
  4. Replicate to 3 channels (DINOv2 RGB input).

Output shape per axis: [3, img_size, img_size]  float32 in [0, 1]
"""

import numpy as np
from typing import Dict, Tuple
from PIL import Image


# axis index: 0=sagittal, 1=coronal, 2=axial
AXIS_IDX = {"sagittal": 0, "coronal": 1, "axial": 2}


def _norm(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)


def _resize(arr2d: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray((arr2d * 255).clip(0, 255).astype(np.uint8), mode="L")
    return np.array(img.resize((size, size), Image.BILINEAR)) / 255.0


def find_max_tumor_slice(mask: np.ndarray, axis: int) -> int:
    areas = mask.sum(axis=tuple(i for i in range(3) if i != axis))
    return int(np.argmax(areas))


def build_montage(
    volumes: Dict[str, np.ndarray],
    mask: np.ndarray,
    modalities: Tuple[str, ...],
    axis_name: str,
    img_size: int,
) -> np.ndarray:
    """
    Returns [3, img_size, img_size] float32 montage for one axis.
    Layout (2×2):  | mod0 | mod1 |
                   | mod2 | mod3 |
    """
    axis_idx = AXIS_IDX[axis_name]
    n = find_max_tumor_slice(mask, axis_idx)
    tile = img_size // 2

    canvas = np.zeros((img_size, img_size), dtype=np.float32)
    positions = [(0, 0), (0, tile), (tile, 0), (tile, tile)]

    for (r, c), mod in zip(positions, modalities):
        sl = np.take(volumes[mod], n, axis=axis_idx)
        canvas[r:r+tile, c:c+tile] = _resize(_norm(sl), tile)

    rgb = np.stack([canvas, canvas, canvas], axis=0)   # [3, H, W]
    return rgb.astype(np.float32)


class MRISlicer:
    def __init__(self, modalities=("T1", "T1ce", "T2", "FLAIR"), img_size=224):
        assert len(modalities) == 4
        self.modalities = modalities
        self.img_size = img_size

    def slice_patient(
        self,
        volumes: Dict[str, np.ndarray],   # {mod: (X,Y,Z) float32}
        mask: np.ndarray,                  # (X,Y,Z) binary
    ) -> Dict[str, np.ndarray]:
        """Returns {"axial": [3,H,W], "coronal": ..., "sagittal": ...}"""
        return {
            ax: build_montage(volumes, mask, self.modalities, ax, self.img_size)
            for ax in AXIS_IDX
        }


def load_nifti(path: str) -> np.ndarray:
    import nibabel as nib
    return nib.load(path).get_fdata().astype(np.float32)
