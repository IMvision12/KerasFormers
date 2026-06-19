GROUNDING_DINO_CONFIG = {
    "grounding_dino_tiny": {
        "d_model": 256,
        "encoder_layers": 6,
        "encoder_ffn_dim": 2048,
        "encoder_attention_heads": 8,
        "decoder_layers": 6,
        "decoder_ffn_dim": 2048,
        "decoder_attention_heads": 8,
        "num_queries": 900,
        "num_feature_levels": 4,
        "encoder_n_points": 4,
        "decoder_n_points": 4,
        "max_text_len": 256,
        "query_dim": 4,
        "two_stage": True,
        "positional_embedding_temperature": 20.0,
        "layer_norm_eps": 1e-5,
        "activation_function": "relu",
        "backbone_embed_dim": 96,
        "backbone_depths": (2, 2, 6, 2),
        "backbone_num_heads": (3, 6, 12, 24),
        "backbone_window_size": 7,
        "backbone_out_indices": (2, 3, 4),
        "text_vocab_size": 30522,
        "text_hidden_size": 768,
        "text_num_layers": 12,
        "text_num_heads": 12,
        "text_intermediate_size": 3072,
        "text_max_position_embeddings": 512,
        "text_layer_norm_eps": 1e-12,
    },
    "grounding_dino_base": {
        "d_model": 256,
        "encoder_layers": 6,
        "encoder_ffn_dim": 2048,
        "encoder_attention_heads": 8,
        "decoder_layers": 6,
        "decoder_ffn_dim": 2048,
        "decoder_attention_heads": 8,
        "num_queries": 900,
        "num_feature_levels": 4,
        "encoder_n_points": 4,
        "decoder_n_points": 4,
        "max_text_len": 256,
        "query_dim": 4,
        "two_stage": True,
        "positional_embedding_temperature": 20.0,
        "layer_norm_eps": 1e-5,
        "activation_function": "relu",
        "backbone_embed_dim": 128,
        "backbone_depths": (2, 2, 18, 2),
        "backbone_num_heads": (4, 8, 16, 32),
        "backbone_window_size": 12,
        "backbone_out_indices": (2, 3, 4),
        "text_vocab_size": 30522,
        "text_hidden_size": 768,
        "text_num_layers": 12,
        "text_num_heads": 12,
        "text_intermediate_size": 3072,
        "text_max_position_embeddings": 512,
        "text_layer_norm_eps": 1e-12,
    },
}

GROUNDING_DINO_WEIGHTS_URLS = {
    "grounding_dino_tiny": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/tag/grounding_dino/grounding_dino_tiny.weights.h5"
    },
    "grounding_dino_base": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/tag/grounding_dino/grounding_dino_base.weights.h5"
    },
}

GROUNDING_DINO_TOKENIZER_URLS = {
    "grounding_dino_tiny": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/grounding_dino/grounding_dino_tiny_tokenizer.json"
    },
    "grounding_dino_base": {
        "tokenizer_json": "https://github.com/IMvision12/KerasFormers/releases/download/grounding_dino/grounding_dino_base_tokenizer.json"
    },
}
