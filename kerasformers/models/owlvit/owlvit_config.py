OWLVIT_CONFIG = {
    "owlvit-base-patch32": {
        "vision_image_size": 768,
        "vision_patch_size": 32,
        "vision_hidden_dim": 768,
        "vision_intermediate_size": 3072,
        "vision_num_layers": 12,
        "vision_num_heads": 12,
        "text_hidden_dim": 512,
        "text_intermediate_size": 2048,
        "text_num_heads": 8,
        "projection_dim": 512,
    },
    "owlvit-base-patch16": {
        "vision_image_size": 768,
        "vision_patch_size": 16,
        "vision_hidden_dim": 768,
        "vision_intermediate_size": 3072,
        "vision_num_layers": 12,
        "vision_num_heads": 12,
        "text_hidden_dim": 512,
        "text_intermediate_size": 2048,
        "text_num_heads": 8,
        "projection_dim": 512,
    },
    "owlvit-large-patch14": {
        "vision_image_size": 840,
        "vision_patch_size": 14,
        "vision_hidden_dim": 1024,
        "vision_intermediate_size": 4096,
        "vision_num_layers": 24,
        "vision_num_heads": 16,
        "text_hidden_dim": 768,
        "text_intermediate_size": 3072,
        "text_num_heads": 16,
        "projection_dim": 768,
    },
}

OWLVIT_WEIGHTS_URLS = {
    "owlvit-base-patch32": {
        "url": "https://huggingface.co/kerasformers/owlvit-base-patch32",
    },
    "owlvit-base-patch16": {
        "url": "https://huggingface.co/kerasformers/owlvit-base-patch16",
    },
    "owlvit-large-patch14": {
        "url": "https://huggingface.co/kerasformers/owlvit-large-patch14",
    },
}
