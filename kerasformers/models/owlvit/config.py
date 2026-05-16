OWLVIT_CONFIG = {
    "owlvit-base-patch32": {
        "vision_image_size": 768,
        "vision_patch_size": 32,
        "vision_hidden_size": 768,
        "vision_intermediate_size": 3072,
        "vision_num_hidden_layers": 12,
        "vision_num_attention_heads": 12,
        "text_hidden_size": 512,
        "text_intermediate_size": 2048,
        "text_num_attention_heads": 8,
        "projection_dim": 512,
    },
    "owlvit-base-patch16": {
        "vision_image_size": 768,
        "vision_patch_size": 16,
        "vision_hidden_size": 768,
        "vision_intermediate_size": 3072,
        "vision_num_hidden_layers": 12,
        "vision_num_attention_heads": 12,
        "text_hidden_size": 512,
        "text_intermediate_size": 2048,
        "text_num_attention_heads": 8,
        "projection_dim": 512,
    },
    "owlvit-large-patch14": {
        "vision_image_size": 840,
        "vision_patch_size": 14,
        "vision_hidden_size": 1024,
        "vision_intermediate_size": 4096,
        "vision_num_hidden_layers": 24,
        "vision_num_attention_heads": 16,
        "text_hidden_size": 768,
        "text_intermediate_size": 3072,
        "text_num_attention_heads": 16,
        "projection_dim": 768,
    },
}

OWLVIT_WEIGHTS = {
    "owlvit-base-patch32": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/owlvit/owlvit_base_patch32.weights.h5",
    },
    "owlvit-base-patch16": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/owlvit/owlvit_base_patch16.weights.h5",
    },
    "owlvit-large-patch14": {
        "url": "https://github.com/IMvision12/KerasFormers/releases/download/owlvit/owlvit_large_patch14.weights.h5",
    },
}
