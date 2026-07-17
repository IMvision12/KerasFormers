import keras
import numpy as np
from keras import ops

from kerasformers.base import BaseImageProcessor

from .qwen2_vl_image_processor import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, smart_resize


@keras.saving.register_keras_serializable(package="kerasformers")
class Qwen2VLVideoProcessor(BaseImageProcessor):
    """Turn videos into ``{"pixel_values_videos", "video_grid_thw"}`` (pure keras.ops).

    Mirrors HF ``Qwen2VLVideoProcessor``: optionally subsample frames to a target
    ``fps`` (``do_sample_frames``), smart-resize every frame to a multiple of
    ``patch_size * spatial_merge_size``, rescale to ``[0, 1]`` and CLIP-normalize,
    pad the frame count up to a multiple of ``temporal_patch_size`` (repeating the
    last frame), and flatten into
    ``(grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size**2)``
    patches in spatial-merge-block order: the exact patch layout the image
    processor emits, so the shared vision tower consumes images and video frames
    identically. ``grid_t = num_frames // temporal_patch_size`` (vs. 1 for images).

    A single video is a ``(num_frames, H, W, C)`` / ``(num_frames, C, H, W)`` array
    or a list of per-frame arrays / PIL images; ``__call__`` also accepts a list of
    such videos plus parallel ``video_metadata`` dicts (``{"fps": ...}``) that drive
    sampling. Pixel values are assumed to be in ``[0, 255]``. Qwen2-VL does not
    sample by default; Qwen3-VL does (2 fps).
    """

    def __init__(
        self,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        min_pixels=128 * 28 * 28,
        max_pixels=768 * 28 * 28,
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
        do_sample_frames=False,
        fps=None,
        num_frames=None,
        min_frames=4,
        max_frames=768,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_mean = ops.cast(ops.convert_to_tensor(image_mean), "float32")
        self.image_std = ops.cast(ops.convert_to_tensor(image_std), "float32")
        self.do_sample_frames = do_sample_frames
        self.fps = fps
        self.num_frames = num_frames
        self.min_frames = min_frames
        self.max_frames = max_frames

    def _as_video_list(self, videos):
        from PIL import Image

        if isinstance(videos, (list, tuple)) and len(videos) > 0:
            first = videos[0]
            if isinstance(first, (list, tuple)) or isinstance(first, Image.Image):
                return [videos] if isinstance(first, Image.Image) else list(videos)
            arr0 = np.asarray(first)
            return list(videos) if arr0.ndim == 4 else [videos]
        arr = np.asarray(videos)
        return [videos] if arr.ndim == 4 else list(videos)

    def _to_thwc(self, video):
        from PIL import Image

        if isinstance(video, (list, tuple)):
            frames = [
                np.asarray(f.convert("RGB"))
                if isinstance(f, Image.Image)
                else np.asarray(f)
                for f in video
            ]
            arr = np.stack(frames, axis=0)
        else:
            arr = np.asarray(video)
        x = ops.cast(ops.convert_to_tensor(arr), "float32")
        if int(x.shape[-1]) != 3 and int(x.shape[1]) == 3:
            x = ops.transpose(x, (0, 2, 3, 1))
        channels = int(x.shape[-1])
        if channels == 1:
            x = ops.repeat(x, 3, axis=-1)
        elif channels == 4:
            x = x[..., :3]
        return x

    def _sample_indices(self, total_num_frames, video_fps):
        """Uniform frame indices for the target fps (HF ``sample_frames``).

        ``num_frames = total / video_fps * self.fps`` clamped to
        ``[min_frames, max_frames, total]``, then ``linspace`` over the clip.
        """
        num_frames = self.num_frames
        if num_frames is None and self.fps is not None:
            if video_fps is None:
                video_fps = 24
            num_frames = int(total_num_frames / video_fps * self.fps)
            num_frames = min(
                max(num_frames, self.min_frames), self.max_frames, total_num_frames
            )
        if num_frames is None:
            num_frames = min(max(total_num_frames, self.min_frames), self.max_frames)
        return np.linspace(0, total_num_frames - 1, num_frames).round().astype("int64")

    def _resized_hw(self, num_frames, h, w):
        """Target (height, width) for each frame. Qwen2-VL resizes per frame; the
        Qwen3-VL subclass overrides this with a frame-count-aware budget."""
        factor = self.patch_size * self.spatial_merge_size
        return smart_resize(h, w, factor, self.min_pixels, self.max_pixels)

    def _preprocess_one(self, video, metadata=None):
        frames = self._to_thwc(video)
        video_fps = metadata.get("fps") if metadata else None
        if self.do_sample_frames and (self.fps is not None or self.num_frames):
            idx = self._sample_indices(int(frames.shape[0]), video_fps)
            frames = ops.take(frames, ops.convert_to_tensor(idx), axis=0)
        num_frames = int(frames.shape[0])
        h, w = int(frames.shape[1]), int(frames.shape[2])
        rh, rw = self._resized_hw(num_frames, h, w)
        frames = ops.image.resize(
            frames, (rh, rw), interpolation="bicubic", antialias=True
        )
        x = frames / 255.0
        x = (x - self.image_mean) / self.image_std

        t = self.temporal_patch_size
        if num_frames % t != 0:
            pad = t - (num_frames % t)
            x = ops.concatenate([x, ops.repeat(x[-1:], pad, axis=0)], axis=0)
        num_frames = int(x.shape[0])

        m, p = self.spatial_merge_size, self.patch_size
        grid_t, grid_h, grid_w = num_frames // t, rh // p, rw // p
        x = ops.transpose(x, (0, 3, 1, 2))
        patches = ops.reshape(x, (grid_t, t, 3, grid_h // m, m, p, grid_w // m, m, p))
        patches = ops.transpose(patches, (0, 3, 6, 4, 7, 2, 1, 5, 8))
        flat = ops.reshape(patches, (grid_t * grid_h * grid_w, 3 * t * p * p))
        return ops.cast(flat, "float32"), [grid_t, grid_h, grid_w]

    def call(self, videos, video_metadata=None):
        vids = self._as_video_list(videos)
        if video_metadata is None:
            metas = [None] * len(vids)
        elif not isinstance(video_metadata, (list, tuple)):
            metas = [video_metadata]
        else:
            metas = list(video_metadata)
        all_patches, grids = [], []
        for video, meta in zip(vids, metas):
            flat, grid = self._preprocess_one(video, meta)
            all_patches.append(flat)
            grids.append(grid)
        return {
            "pixel_values_videos": ops.concatenate(all_patches, axis=0),
            "video_grid_thw": np.array(grids, dtype="int64"),
        }
