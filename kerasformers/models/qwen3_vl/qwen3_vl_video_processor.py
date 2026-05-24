import math

from kerasformers.models.qwen2_vl.qwen2_vl_video_processor import Qwen2VLVideoProcessor


def qwen3_smart_resize(
    num_frames, height, width, temporal_factor, factor, min_pixels, max_pixels
):
    """Frame-count-aware smart resize (HF ``Qwen3VLVideoProcessor.smart_resize``).

    Unlike Qwen2-VL's per-frame resize, the pixel budget covers the whole clip:
    the rounded ``t_bar * h_bar * w_bar`` token volume is clamped into
    ``[min_pixels, max_pixels]``, so more frames -> smaller frames.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be >= factor:{factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = math.ceil(num_frames / temporal_factor) * temporal_factor
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class Qwen3VLVideoProcessor(Qwen2VLVideoProcessor):
    """Qwen3-VL video processor: like :class:`Qwen2VLVideoProcessor` but with a
    16px patch, ``[0.5, 0.5, 0.5]`` mean/std, and a clip-level (frame-count-aware)
    resize budget. The flattened patch layout is identical, so the shared vision
    tower consumes the output unchanged. Pixel values are assumed in ``[0, 255]``.
    """

    def __init__(
        self,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        min_pixels=128 * 32 * 32,
        max_pixels=768 * 32 * 32,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        do_sample_frames=True,
        fps=2,
        num_frames=None,
        min_frames=4,
        max_frames=768,
    ):
        super().__init__(
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            temporal_patch_size=temporal_patch_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            image_mean=image_mean,
            image_std=image_std,
            do_sample_frames=do_sample_frames,
            fps=fps,
            num_frames=num_frames,
            min_frames=min_frames,
            max_frames=max_frames,
        )

    def _resized_hw(self, num_frames, h, w):
        factor = self.patch_size * self.spatial_merge_size
        return qwen3_smart_resize(
            num_frames,
            h,
            w,
            self.temporal_patch_size,
            factor,
            self.min_pixels,
            self.max_pixels,
        )
