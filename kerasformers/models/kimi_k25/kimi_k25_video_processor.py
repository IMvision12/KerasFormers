import keras
import numpy as np
from PIL import Image

from kerasformers.base import BaseImageProcessor

from .kimi_k25_image_processor import patchify, resize_and_pad


@keras.saving.register_keras_serializable(package="kerasformers")
class KimiK25VideoProcessor(BaseImageProcessor):
    """Chunked video preprocessor for Kimi K2.5's MoonViT.

    A video is split into chunks of ``temporal_patch_size`` (4) frames -- the
    vision tower's temporal position table holds exactly that many -- and each
    chunk becomes one ``(t, h, w)`` clip that the tower averages over time. Frames
    are resized under a *per-frame* patch budget (4096, tighter than the still
    image's 16384), zero-padded to a whole merge cell, normalized, and flattened
    frame-major into ``(t * grid_h * grid_w, 3, patch, patch)``.

    Returns ``num_chunks_per_video`` alongside the patches so the processor can
    lay out one media span per chunk.

    Args:
        patch_size: MoonViT patch (14).
        merge_size: Spatial patch-merge kernel (2).
        temporal_patch_size: Frames per chunk (4).
        max_patches: Per-frame patch budget (4096).
        max_side: Per-side patch budget (512).
        image_mean / image_std: Per-channel normalization.
    """

    def __init__(
        self,
        patch_size=14,
        merge_size=2,
        temporal_patch_size=4,
        max_patches=4096,
        max_side=512,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size
        self.max_patches = max_patches
        self.max_side = max_side
        self.image_mean = tuple(image_mean)
        self.image_std = tuple(image_std)

    @classmethod
    def from_hf(cls, repo, **kwargs):
        import json

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo.removeprefix("hf:"), "preprocessor_config.json")
        with open(path) as handle:
            cfg = json.load(handle).get("media_proc_cfg", {})
        defaults = {
            "patch_size": cfg.get("patch_size", 14),
            "merge_size": cfg.get("merge_kernel_size", 2),
            "temporal_patch_size": cfg.get("temporal_merge_kernel_size", 4),
            "max_patches": cfg.get("in_patch_limit_each_frame", 4096),
            "max_side": cfg.get("patch_limit_on_one_side", 512),
            "image_mean": tuple(cfg.get("image_mean", (0.5, 0.5, 0.5))),
            "image_std": tuple(cfg.get("image_std", (0.5, 0.5, 0.5))),
        }
        defaults.update(kwargs)
        return cls(**defaults)

    def normalize(self, image):
        mean = np.array(self.image_mean, dtype="float32")[:, None, None]
        std = np.array(self.image_std, dtype="float32")[:, None, None]
        array = np.asarray(image, dtype="float32").transpose(2, 0, 1) / 255.0
        return (array - mean) / std

    def process_chunk(self, frames):
        patches, grid = [], None
        for frame in frames:
            frame = resize_and_pad(
                frame.convert("RGB"),
                self.patch_size,
                self.merge_size,
                self.max_patches,
                self.max_side,
            )
            frame_patches, grid = patchify(self.normalize(frame), self.patch_size)
            patches.append(frame_patches)
        return np.concatenate(patches, axis=0), (len(frames), *grid)

    def call(self, videos):
        if isinstance(videos, Image.Image):
            videos = [[videos]]
        elif videos and isinstance(videos[0], Image.Image):
            videos = [videos]
        pixel_values, grids, chunks_per_video = [], [], []
        for frames in videos:
            frames = list(frames)
            step = self.temporal_patch_size
            chunks = [frames[i : i + step] for i in range(0, len(frames), step)]
            chunks_per_video.append(len(chunks))
            for chunk in chunks:
                patches, grid = self.process_chunk(chunk)
                pixel_values.append(patches)
                grids.append(grid)
        return {
            "pixel_values_videos": np.concatenate(pixel_values, axis=0).astype(
                "float32"
            ),
            "video_grid_thw": np.array(grids, dtype="int32"),
            "num_chunks_per_video": chunks_per_video,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "merge_size": self.merge_size,
                "temporal_patch_size": self.temporal_patch_size,
                "max_patches": self.max_patches,
                "max_side": self.max_side,
                "image_mean": self.image_mean,
                "image_std": self.image_std,
            }
        )
        return config
