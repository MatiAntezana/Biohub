"""Read-only TZYX image access and measured (never zero-filled) HOCT features."""
from __future__ import annotations
from collections import OrderedDict

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


class ImageSource:
    def __init__(self, path, key, scale_um):
        import zarr
        group = zarr.open_group(str(path), mode="r")
        self.array = group[key]
        self.shape = tuple(self.array.shape)
        self.scale = np.asarray(scale_um, float)
        if len(self.shape) != 4:
            raise ValueError("Only explicit TZYX images are supported; do not guess channel axes")
        metadata = group.attrs.get("multiscales", [])
        if not metadata:
            raise ValueError("NGFF axes/scale metadata required")
        axes = metadata[0]["axes"]
        # OME-NGFF permits axis labels such as T/Z/Y/X.  Interpret only the
        # spelling case-insensitively; axis order, units and scale remain exact.
        axis_names = [(a["name"] if isinstance(a, dict) else a).lower() for a in axes]
        if axis_names != ["t", "z", "y", "x"]:
            raise ValueError("NGFF axes must be t,z,y,x")
        for axis in axes[1:]:
            if not isinstance(axis, dict) or axis.get("unit") not in {"micrometer", "micrometre", "µm"}:
                raise ValueError("NGFF spatial units must explicitly be micrometres")
        entries = [d for d in metadata[0]["datasets"] if d["path"] == key]
        if len(entries) != 1:
            raise ValueError("Ambiguous NGFF resolution level")
        transforms = entries[0].get("coordinateTransformations", []) + metadata[0].get("coordinateTransformations", [])
        scale = np.ones(4)
        for transform in transforms:
            if transform["type"] == "scale":
                scale *= np.asarray(transform["scale"])
            elif transform["type"] != "translation" or any(transform["translation"]):
                raise ValueError("Unsupported nonzero NGFF coordinate transform")
        if not np.allclose(scale[1:], self.scale, rtol=1e-5):
            raise ValueError("Manifest/image voxel spacing mismatch")
        self.cache = OrderedDict()

    def frame(self, t):
        t = int(t)
        if t not in range(self.shape[0]):
            raise IndexError("Frame outside image")
        if t not in self.cache:
            image = np.asarray(self.array[t], dtype=np.float32)
            if not np.isfinite(image).all():
                raise ValueError("Image contains non-finite intensity")
            low, high = float(image.min()), float(np.quantile(image, .999))
            if high <= low:
                raise ValueError(f"No measurable contrast at frame {t}")
            self.cache[t] = (image - low) / (high - low)
        self.cache.move_to_end(t)
        while len(self.cache) > 4:
            self.cache.popitem(last=False)
        return self.cache[t]

    def patch(self, t, centre, half_um=8.0, shape=(9, 25, 25)):
        """Fixed physical field of view, sampled in image pixels; no full movie load."""
        grid = np.meshgrid(*[np.linspace(-half_um / s, half_um / s, size) + c
                              for s, size, c in zip(self.scale, shape, centre)], indexing="ij")
        return ndimage.map_coordinates(self.frame(t), grid, order=1, mode="constant", cval=0).astype(np.float32)


def measured_features(coords, source, radius_um=5.0, min_voxels=4):
    """HOCT's 19-column order, from connected local foreground and Voronoi ownership.

    Masks are an experimental point-to-image bridge, not a pretrained segmenter.
    Pixel moments/positions match the upstream feature convention; micrometres
    are used separately for candidate neighbourhoods. Detection positions stay fixed.
    """
    output = np.empty((len(coords), 19), dtype=np.float32)
    shape = np.asarray(source.shape[1:])
    half = np.ceil(radius_um / source.scale).astype(int)
    for t in np.unique(coords[:, 0]).astype(int):
        ids = np.flatnonzero(coords[:, 0] == t)
        frame = source.frame(t)
        tree = cKDTree(coords[ids, 1:] * source.scale)
        for local_id, node in enumerate(ids):
            centre = coords[node, 1:]
            start = np.maximum(0, np.floor(centre).astype(int) - half)
            stop = np.minimum(shape, np.ceil(centre).astype(int) + half + 1)
            crop = frame[tuple(slice(a, b) for a, b in zip(start, stop))]
            grid = np.indices(crop.shape).reshape(3, -1).T + start
            distance = np.linalg.norm((grid - centre) * source.scale, axis=1)
            owner = tree.query(grid * source.scale, k=1)[1]
            near = (distance <= radius_um) & (owner == local_id)
            seed_area = near & (distance <= max(1.5, float(source.scale.max())))
            values = crop.ravel()
            if not seed_area.any():
                raise ValueError(f"No unambiguous image seed for detection {node}")
            seed = np.flatnonzero(seed_area)[np.argmax(values[seed_area])]
            background = float(np.median(values))
            peak = float(values[seed])
            threshold = background + .35 * (peak - background)
            mask = (near & (values > threshold)).reshape(crop.shape)
            labels, _ = ndimage.label(mask)
            component = labels.ravel()[seed]
            selected = np.flatnonzero(labels.ravel() == component) if component else np.array([], int)
            if peak <= background or len(selected) < min_voxels:
                raise ValueError(f"Insufficient image-derived mask at node {node}; no fabricated HOCT features")
            points = grid[selected].astype(float)
            centred = points - points.mean(axis=0)
            covariance = centred.T @ centred / len(points)
            inertia = np.eye(3) * np.trace(covariance) - covariance
            signal = values[selected]
            diameter = (6 * len(points) / np.pi) ** (1 / 3)
            border = 1 - min(1.0, float(np.minimum(centre, shape - centre).min()) / 5)
            output[node] = [*coords[node], diameter, signal.min(), signal.max(), signal.mean(),
                            signal.std(), *inertia.ravel(), border]
    if not np.isfinite(output).all():
        raise ValueError("Non-finite image features")
    return output


def normalized_correlation(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 1e-8 else -1.0
