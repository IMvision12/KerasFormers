import math

import keras
import numpy as np
from keras import ops

from kerasformers.base import BaseAudioFeatureExtractor


def hz_to_mel_htk(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def mel_to_hz_htk(mels):
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def build_mel_filter_bank_htk(n_fft, n_mels, sample_rate, f_min=0.0, f_max=None):
    # Matches torchaudio.functional.melscale_fbanks with mel_scale="htk", norm=None
    # (the defaults used by GraniteSpeechFeatureExtractor's MelSpectrogram).
    f_max = f_max if f_max is not None else sample_rate / 2.0
    n_freqs = n_fft // 2 + 1
    all_freqs = np.linspace(0, sample_rate / 2.0, n_freqs)

    m_min = hz_to_mel_htk(f_min)
    m_max = hz_to_mel_htk(f_max)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz_htk(m_pts)

    f_diff = np.diff(f_pts)
    slopes = f_pts[None, :] - all_freqs[:, None]  # (n_freqs, n_mels+2)
    down = -slopes[:, :-2] / f_diff[:-1][None, :]
    up = slopes[:, 2:] / f_diff[1:][None, :]
    fb = np.maximum(0.0, np.minimum(down, up))  # (n_freqs, n_mels)
    return fb.astype("float32")


@keras.saving.register_keras_serializable(package="kerasformers")
class GraniteSpeechFeatureExtractor(BaseAudioFeatureExtractor):
    """Mel-spectrogram feature extractor for Granite Speech (pure Keras 3).

    Reproduces HF ``GraniteSpeechFeatureExtractor``:

    * 16 kHz, ``torchaudio``-style ``MelSpectrogram`` (n_fft=512, win=400,
      hop=160, 80 HTK mel bins, power spectrogram), centered reflect-padded STFT.
    * ``log10`` of the (transposed) mel, clamped at ``max - 8.0``, then ``/4 + 1``.
    * the last frame is dropped if the frame count is odd, and consecutive frames
      are stacked in pairs -> ``input_features`` of width ``2 * n_mels``.

    ``call`` also returns ``audio_embed_sizes`` (the projector output length per
    clip) and a boolean ``input_features_mask`` over the padded projector tokens,
    mirroring the HF processor contract.
    """

    model_input_names = ["input_features"]

    def __init__(
        self,
        sampling_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=80,
        projector_window_size=15,
        projector_downsample_rate=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.projector_window_size = projector_window_size
        self.projector_downsample_rate = projector_downsample_rate
        self.mel_filters = build_mel_filter_bank_htk(n_fft, n_mels, sampling_rate)

    def normalize_waves(self, audios):
        if isinstance(audios, np.ndarray):
            waves = [audios] if audios.ndim == 1 else list(audios)
        elif isinstance(audios, (list, tuple)):
            waves = [np.asarray(w, dtype=np.float32).squeeze() for w in audios]
        else:
            arr = np.asarray(audios, dtype=np.float32).squeeze()
            waves = [arr]
        lengths = [int(np.asarray(w).shape[-1]) for w in waves]
        max_len = max(lengths)
        out = np.zeros((len(waves), max_len), dtype=np.float32)
        for i, w in enumerate(waves):
            w = np.asarray(w, dtype=np.float32).reshape(-1)
            out[i, : len(w)] = w
        return out, lengths

    def log_mel(self, batch):
        # win_length (400) < n_fft (512): torchaudio centers the Hann window in the
        # n_fft frame (zero-padded both sides), so build that padded window.
        pad = (self.n_fft - self.win_length) // 2
        hann = ops.convert_to_tensor(
            np.hanning(self.win_length + 1)[:-1].astype("float32")
        )
        window = ops.pad(hann, [[pad, self.n_fft - self.win_length - pad]])

        real, imag = ops.stft(
            batch,
            sequence_length=self.n_fft,
            sequence_stride=self.hop_length,
            fft_length=self.n_fft,
            window=window,
            center=True,
        )
        power = real * real + imag * imag  # (B, n_frames, n_fft//2+1)
        mel = ops.matmul(
            power, ops.convert_to_tensor(self.mel_filters)
        )  # (B, n_frames, n_mels)
        # HF: transpose to (B, n_mels, n_frames) then clip+log10, but the downstream
        # stacking reshapes back over (n_frames, n_mels); keep (B, n_frames, n_mels).
        inv_log10 = 1.0 / math.log(10.0)
        logmel = ops.log(ops.maximum(mel, 1e-10)) * inv_log10
        mx = ops.max(logmel, axis=(1, 2), keepdims=True)
        logmel = ops.maximum(logmel, mx - 8.0)
        logmel = logmel / 4.0 + 1.0
        return logmel

    def stack_frames(self, logmel):
        n_frames = int(logmel.shape[1])
        if n_frames % 2 == 1:
            logmel = logmel[:, :-1, :]
            n_frames -= 1
        b = int(logmel.shape[0])
        return ops.reshape(logmel, (b, n_frames // 2, 2 * self.n_mels))

    def num_audio_features(self, audio_lengths):
        eff = self.projector_window_size // self.projector_downsample_rate
        sizes = []
        for raw in audio_lengths:
            mel_len = raw // self.hop_length + 1
            enc_len = mel_len // 2
            nblocks = math.ceil(enc_len / self.projector_window_size)
            sizes.append(nblocks * eff)
        return sizes

    def call(self, raw_speech, sampling_rate=16000):
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"GraniteSpeechFeatureExtractor expects {self.sampling_rate} Hz "
                f"input; got {sampling_rate} Hz."
            )
        batch_np, lengths = self.normalize_waves(raw_speech)
        batch = ops.convert_to_tensor(batch_np, dtype="float32")
        logmel = self.log_mel(batch)
        features = self.stack_frames(logmel)

        embed_sizes = self.num_audio_features(lengths)
        max_size = max(embed_sizes)
        mask = ops.convert_to_tensor(
            np.arange(max_size)[None, :] < np.array(embed_sizes)[:, None]
        )
        return {
            "input_features": features,
            "audio_embed_sizes": embed_sizes,
            "input_features_mask": mask,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sampling_rate": self.sampling_rate,
                "n_fft": self.n_fft,
                "win_length": self.win_length,
                "hop_length": self.hop_length,
                "n_mels": self.n_mels,
                "projector_window_size": self.projector_window_size,
                "projector_downsample_rate": self.projector_downsample_rate,
            }
        )
        return config
