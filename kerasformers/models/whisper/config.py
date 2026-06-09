WHISPER_CONFIG = {
    "whisper_tiny": {
        "hidden_dim": 384,
        "encoder_num_layers": 4,
        "encoder_attention_heads": 6,
        "encoder_ffn_dim": 1536,
        "decoder_num_layers": 4,
        "decoder_attention_heads": 6,
        "decoder_ffn_dim": 1536,
        "vocab_size": 51865,
        "max_source_positions": 1500,
        "max_target_positions": 448,
        "num_mel_bins": 80,
    },
    "whisper_base": {
        "hidden_dim": 512,
        "encoder_num_layers": 6,
        "encoder_attention_heads": 8,
        "encoder_ffn_dim": 2048,
        "decoder_num_layers": 6,
        "decoder_attention_heads": 8,
        "decoder_ffn_dim": 2048,
        "vocab_size": 51865,
        "max_source_positions": 1500,
        "max_target_positions": 448,
        "num_mel_bins": 80,
    },
    "whisper_small": {
        "hidden_dim": 768,
        "encoder_num_layers": 12,
        "encoder_attention_heads": 12,
        "encoder_ffn_dim": 3072,
        "decoder_num_layers": 12,
        "decoder_attention_heads": 12,
        "decoder_ffn_dim": 3072,
        "vocab_size": 51865,
        "max_source_positions": 1500,
        "max_target_positions": 448,
        "num_mel_bins": 80,
    },
    "whisper_medium": {
        "hidden_dim": 1024,
        "encoder_num_layers": 24,
        "encoder_attention_heads": 16,
        "encoder_ffn_dim": 4096,
        "decoder_num_layers": 24,
        "decoder_attention_heads": 16,
        "decoder_ffn_dim": 4096,
        "vocab_size": 51865,
        "max_source_positions": 1500,
        "max_target_positions": 448,
        "num_mel_bins": 80,
    },
    "whisper_large": {
        "hidden_dim": 1280,
        "encoder_num_layers": 32,
        "encoder_attention_heads": 20,
        "encoder_ffn_dim": 5120,
        "decoder_num_layers": 32,
        "decoder_attention_heads": 20,
        "decoder_ffn_dim": 5120,
        "vocab_size": 51865,
        "max_source_positions": 1500,
        "max_target_positions": 448,
        "num_mel_bins": 80,
    },
    "whisper_large_v2": {
        "hidden_dim": 1280,
        "encoder_num_layers": 32,
        "encoder_attention_heads": 20,
        "encoder_ffn_dim": 5120,
        "decoder_num_layers": 32,
        "decoder_attention_heads": 20,
        "decoder_ffn_dim": 5120,
        "vocab_size": 51865,
        "max_source_positions": 1500,
        "max_target_positions": 448,
        "num_mel_bins": 80,
    },
    "whisper_large_v3": {
        "hidden_dim": 1280,
        "encoder_num_layers": 32,
        "encoder_attention_heads": 20,
        "encoder_ffn_dim": 5120,
        "decoder_num_layers": 32,
        "decoder_attention_heads": 20,
        "decoder_ffn_dim": 5120,
        "vocab_size": 51866,
        "max_source_positions": 1500,
        "max_target_positions": 448,
        "num_mel_bins": 128,
    },
    "whisper_large_v3_turbo": {
        "hidden_dim": 1280,
        "encoder_num_layers": 32,
        "encoder_attention_heads": 20,
        "encoder_ffn_dim": 5120,
        "decoder_num_layers": 4,
        "decoder_attention_heads": 20,
        "decoder_ffn_dim": 5120,
        "vocab_size": 51866,
        "max_source_positions": 1500,
        "max_target_positions": 448,
        "num_mel_bins": 128,
    },
}


WHISPER_WEIGHTS_URLS = {
    "whisper_tiny": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/whisper/whispertiny_openai.weights.h5",
    },
    "whisper_base": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/whisper/whisperbase_openai.weights.h5",
    },
    "whisper_small": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/whisper/whispersmall_openai.weights.h5",
    },
    "whisper_medium": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/whisper/whispermedium_openai.weights.json",
    },
    "whisper_large": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/whisper/whisperlarge_openai.weights.json",
    },
    "whisper_large_v2": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/whisper/whisperlargev2_openai.weights.json",
    },
    "whisper_large_v3": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/whisper/whisperlargev3_openai.weights.json",
    },
    "whisper_large_v3_turbo": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/whisper/whisperlargev3turbo_openai.weights.json",
    },
}


# OpenAI Whisper's hard-coded logits processor lists (see
# `whisper/decoding.py` and the reference `GenerationConfig`).
WHISPER_SUPPRESS_TOKENS = [
    1,
    2,
    7,
    8,
    9,
    10,
    14,
    25,
    26,
    27,
    28,
    29,
    31,
    58,
    59,
    60,
    61,
    62,
    63,
    90,
    91,
    92,
    93,
    359,
    503,
    522,
    542,
    873,
    893,
    902,
    918,
    922,
    931,
    1350,
    1853,
    1982,
    2460,
    2627,
    3246,
    3253,
    3268,
    3536,
    3846,
    3961,
    4183,
    4667,
    6585,
    6647,
    7273,
    9061,
    9383,
    10428,
    10929,
    11938,
    12033,
    12331,
    12562,
    13793,
    14157,
    14635,
    15265,
    15618,
    16553,
    16604,
    18362,
    18956,
    20075,
    21675,
    22520,
    26130,
    26161,
    26435,
    28279,
    29464,
    31650,
    32302,
    32470,
    36865,
    42863,
    47425,
    49870,
    50254,
    50258,
    50360,
    50361,
    50362,
]
WHISPER_BEGIN_SUPPRESS_TOKENS = [220, 50257]

# Per-variant tokenizer.json. The v3 variants (large_v3 / large_v3_turbo) have a
# 51866-token vocab (one extra language) vs 51865 for the rest, so each carries its
# own file.
WHISPER_TOKENIZER_URLS = {
    "whisper_tiny": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/whisper/whisper_tiny_tokenizer.json"
    },
    "whisper_base": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/whisper/whisper_base_tokenizer.json"
    },
    "whisper_small": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/whisper/whisper_small_tokenizer.json"
    },
    "whisper_medium": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/whisper/whisper_medium_tokenizer.json"
    },
    "whisper_large": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/whisper/whisper_large_tokenizer.json"
    },
    "whisper_large_v2": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/whisper/whisper_large_v2_tokenizer.json"
    },
    "whisper_large_v3": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/whisper/whisper_large_v3_tokenizer.json"
    },
    "whisper_large_v3_turbo": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/whisper/whisper_large_v3_turbo_tokenizer.json"
    },
}
