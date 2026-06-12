# KerasFormers 🚀

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Keras](https://img.shields.io/badge/keras-v3.5.0+-success.svg)](https://github.com/keras-team/keras)
![Python](https://img.shields.io/badge/python-v3.10.0+-success.svg)

## 📖 Introduction

KerasFormers is a collection of models with pretrained weights, built entirely with Keras 3. It supports a range of tasks, including classification, object detection (DETR, RT-DETR, RT-DETRv2, RF-DETR, D-FINE, OWL-ViT, OWLv2), segmentation (SAM, SAM2, SAM3, SegFormer, DeepLabV3, EoMT, MaskFormer, Mask2Former, OneFormer, MobileViT-DeepLabV3, RF-DETR), monocular depth estimation (Depth Anything V1, Depth Anything V2), feature extraction (DINO, DINOv2, DINOv3), vision-language modeling (CLIP, SigLIP, SigLIP2, MetaCLIP 2), speech recognition (Whisper, Speech2Text, Moonshine), speech-aware language modeling (Granite Speech, Granite Speech Plus), text encoding and masked language modeling (BERT, RoBERTa, XLM-RoBERTa, DeBERTa, DeBERTa-v2, DeBERTa-v3), text generation with large language models (GPT, GPT-2, Qwen2, Qwen3, Qwen3.5, GPT-OSS, Llama 2, Llama 3, Llama 4, Mistral, Mixtral, Mistral Large 3, Gemma, Gemma 2, Gemma 4, MiniMax-Text-01, MiniMax-M2), multimodal vision-language generation (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, InternVL3, Gemma 3, Mistral 3, DeepSeek-VL, Janus-Pro, MiniMax-M3-VL), and more. It includes hybrid architectures like MaxViT alongside traditional CNNs and pure transformers. kerasformers includes custom layers and backbone support, providing flexibility and efficiency across various applications. For backbones, there are various weight variants like `in1k`, `in21k`, `fb_dist_in1k`, `ms_in22k`, `fb_in22k_ft_in1k`, `ns_jft_in1k`, `aa_in1k`, `cvnets_in1k`, `augreg_in21k_ft_in1k`, `augreg_in21k`, and many more.

## ⚡ Installation

From PyPI (recommended)

```shell
pip install -U kerasformers
```

From Source

```shell
pip install -U git+https://github.com/IMvision12/KerasFormers
```

## 📑 Documentation

Per-model guides - with architecture notes, usage examples, and available pretrained weights, live in the [`docs/`](docs/) folder, one page per model across every supported task (classification, object detection, segmentation, depth estimation, feature extraction, vision-language, speech recognition, text encoding, and language modeling). Classification backbones share a single [page](docs/classification_backbones.md) since they all follow the same `XModel` / `XImageClassify` two-class structure; each other model has its own. Browse [`docs/`](docs/) for the complete, always-up-to-date list.

## 📑 Models

### 📝 Text Models

- Text Encoders (text → embeddings, masked LM, classification)

    | 🏷️ Model Name | 📜 Reference Paper | 📦 Source of Weights |
    |---------------|-------------------|---------------------|
    | BERT | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) | `transformers` |
    | RoBERTa | [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) | `transformers` |
    | XLM-RoBERTa | [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) | `transformers` |
    | DeBERTa | [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) | `transformers` |
    | DeBERTa-v2 | [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) | `transformers` |
    | DeBERTa-v3 | [DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing](https://arxiv.org/abs/2111.09543) | `transformers` |

- Text LLMs (text → text)

    | 🏷️ Model Name | 📜 Reference Paper | 📦 Source of Weights |
    |---------------|-------------------|---------------------|
    | Qwen2 | [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671) | `transformers` |
    | Qwen3 | [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) | `transformers` |
    | Qwen3.5 | [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) | `transformers` |
    | GPT-OSS | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | `transformers` |
    | GPT | [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | `transformers` |
    | GPT-2 | [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | `transformers` |
    | Llama 2 | [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) | `transformers` (gated) |
    | Llama 3 | [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) | `transformers` (gated) |
    | Llama 4 | [meta-llama/Llama-4-Scout-17B-16E](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E) | `transformers` (gated) |
    | Mistral | [Mistral 7B](https://arxiv.org/abs/2310.06825) | `transformers` (gated) |
    | Mixtral | [Mixtral of Experts](https://arxiv.org/abs/2401.04088) | `transformers` (gated) |
    | Mistral Large 3 | [mistralai/Mistral-Large-3-675B-Instruct-2512](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512) | architecture only |
    | Gemma | [Gemma: Open Models Based on Gemini Research and Technology](https://arxiv.org/abs/2403.08295) | `transformers` (gated) |
    | Gemma 2 | [Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) | `transformers` (gated) |
    | Gemma 4 | [google/gemma-4-12B](https://huggingface.co/google/gemma-4-12B) | `transformers` |
    | MiniMax-Text-01 | [MiniMax-01: Scaling Foundation Models with Lightning Attention](https://arxiv.org/abs/2501.08313) | `transformers` |
    | MiniMax-M2 | [MiniMaxAI/MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) | `transformers` |

<br>

### 👁️ Vision Models

- Backbones

    | 🏷️ Model Name | 📜 Reference Paper | 📦 Source of Weights |
    |---------------|-------------------|---------------------|
    | CaiT | [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239) | `timm` |
    | ConvMixer | [Patches Are All You Need?](https://arxiv.org/abs/2201.09792) | `timm` |
    | ConvNeXt | [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) | `timm` |
    | ConvNeXt V2 | [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808) | `timm` |
    | DeiT | [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) | `timm` |
    | DenseNet | [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) | `timm` |
    | EfficientFormer | [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191) | `timm` |
    | EfficientNet | [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) | `timm` |
    | EfficientNet-Lite | [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) | `timm` |
    | EfficientNetV2 | [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) | `timm` |
    | FlexiViT | [FlexiViT: One Model for All Patch Sizes](https://arxiv.org/abs/2212.08013) | `timm` |
    | InceptionNeXt | [InceptionNeXt: When Inception Meets ConvNeXt](https://arxiv.org/abs/2303.16900) | `timm` |
    | Inception-ResNet-v2 | [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261) | `timm` |
    | Inception-v3 | [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) | `timm` |
    | Inception-v4 | [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261) | `timm` |
    | MaxViT | [MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/abs/2204.01697) | `timm` |
    | MiT | [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) | `transformers` |
    | MLP-Mixer | [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601) | `timm` |
    | MobileNetV2 | [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) | `timm` |
    | MobileNetV3 | [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) | `keras` |
    | MobileViT | [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178) | `transformers` |
    | MobileViTV2 | [Separable Self-attention for Mobile Vision Transformers](https://arxiv.org/abs/2206.02680) | `transformers` |
    | NextViT | [Next-ViT: Next Generation Vision Transformer for Efficient Deployment in Realistic Industrial Scenarios](https://arxiv.org/abs/2207.05501) | `timm` |
    | PiT | [Rethinking Spatial Dimensions of Vision Transformers](https://arxiv.org/abs/2103.16302) | `timm` |
    | PoolFormer | [MetaFormer is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418) | `timm` |
    | Res2Net | [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1904.01169) | `timm` |
    | ResMLP | [ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/abs/2105.03404) | `timm` |
    | ResNet | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | `timm` |
    | ResNetV2 | [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) | `timm` |
    | ResNeXt | [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431) | `timm` |
    | SENet | [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) | `timm` |
    | Swin Transformer | [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) | `timm` |
    | Swin Transformer V2 | [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883) | `timm` |
    | VGG | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) | `timm` |
    | ViT | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) | `timm` |
    | Xception | [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) | `timm` |

<br>

- Object Detection

    | 🏷️ Model Name | 📜 Reference Paper | 📦 Source of Weights |
    |---------------|-------------------|---------------------|
    | D-FINE | [D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement](https://arxiv.org/abs/2410.13842) | `transformers` |
    | DETR | [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) | `transformers` |
    | RT-DETR | [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069) | `transformers` |
    | RT-DETRv2 | [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformers](https://arxiv.org/abs/2407.17140) | `transformers` |
    | RF-DETR | [RF-DETR: Neural Architecture Search for Real-Time Detection Transformers](https://arxiv.org/abs/2511.09554) | `transformers` |
    | OWL-ViT | [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230) | `transformers` |
    | OWLv2 | [Scaling Open-Vocabulary Object Detection](https://arxiv.org/abs/2306.09683) | `transformers` |

<br>

- Segmentation

    | 🏷️ Model Name | 📜 Reference Paper | 📦 Source of Weights |
    |---------------|-------------------|---------------------|
    | DeepLabV3 | [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587) | `torchvision` |
    | EoMT | [Your ViT is Secretly an Image Segmentation Model](https://arxiv.org/abs/2503.19108) | `transformers` |
    | MaskFormer | [Per-Pixel Classification is Not All You Need for Semantic Segmentation](https://arxiv.org/abs/2107.06278) | `transformers` |
    | Mask2Former | [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527) | `transformers` |
    | MobileViT | [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178) | `transformers` |
    | MobileViTV2 | [Separable Self-attention for Mobile Vision Transformers](https://arxiv.org/abs/2206.02680) | `transformers` |
    | OneFormer | [OneFormer: One Transformer to Rule Universal Image Segmentation](https://arxiv.org/abs/2211.06220) | `transformers` |
    | RF-DETR | [RF-DETR: Neural Architecture Search for Real-Time Detection Transformers](https://arxiv.org/abs/2511.09554) | `transformers` |
    | SAM | [Segment Anything](https://arxiv.org/abs/2304.02643) | `transformers` |
    | SAM2 | [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) | `transformers` |
    | SAM3 | [SAM 3: Segment Anything with Concepts](https://arxiv.org/abs/2511.16719) | `transformers` (gated) |
    | SegFormer | [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) | `transformers` |

<br>

- Feature Extraction

    | 🏷️ Model Name | 📜 Reference Paper | 📦 Source of Weights |
    |---------------|-------------------|---------------------|
    | DINO | [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294) | `torch.hub` |
    | DINOv2 | [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) | `transformers` |
    | DINOv3 | [DINOv3: Self-Supervised Visual Representation Learning at Scale](https://arxiv.org/abs/2508.10104) | `transformers` (gated) |

<br>

- Depth Estimation

    | 🏷️ Model Name | 📜 Reference Paper | 📦 Source of Weights |
    |---------------|-------------------|---------------------|
    | Depth Anything V1 | [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](https://arxiv.org/abs/2401.10891) | `transformers` |
    | Depth Anything V2 | [Depth Anything V2](https://arxiv.org/abs/2406.09414) | `transformers` |

<br>

### 🖼️ Multimodal Models

- Vision-Language Encoders

    | 🏷️ Model Name | 📜 Reference Paper | 📦 Source of Weights |
    |---------------|-------------------|---------------------|
    | CLIP | [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) | `transformers` |
    | MetaCLIP 2 | [MetaCLIP 2: A Worldwide Scaling Recipe](https://arxiv.org/abs/2507.22062) | `transformers` |
    | SigLIP | [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) | `transformers` |
    | SigLIP2 | [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786) | `transformers` |

<br>

- Multimodal LLMs (image + text → text)

    | 🏷️ Model Name | 📜 Reference Paper | 📦 Source of Weights |
    |---------------|-------------------|---------------------|
    | Qwen2-VL | [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191) | on-the-fly `hf:` |
    | Qwen2.5-VL | [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923) | on-the-fly `hf:` |
    | Qwen3-VL | [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) | on-the-fly `hf:` |
    | InternVL3 | [InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models](https://arxiv.org/abs/2504.10479) | on-the-fly `hf:` |
    | Gemma 3 | [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786) | on-the-fly `hf:` (gated) |
    | Mistral 3 | [mistralai/Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) | on-the-fly `hf:` |
    | DeepSeek-VL | [DeepSeek-VL: Towards Real-World Vision-Language Understanding](https://arxiv.org/abs/2403.05525) | on-the-fly `hf:` |
    | Janus-Pro | [Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling](https://arxiv.org/abs/2501.17811) | on-the-fly `hf:` |
    | MiniMax-M3-VL | [MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) | on-the-fly `hf:` |

<br>

### 🔊 Audio Models

- Speech (speech → text)

    | 🏷️ Model Name | 📜 Reference Paper | 📦 Source of Weights |
    |---------------|-------------------|---------------------|
    | Whisper | [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) | `transformers` |
    | Speech2Text | [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://arxiv.org/abs/2010.05171) | `transformers` |
    | Moonshine | [Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) | `transformers` |

<br>

- Speech LLMs (audio + text → text)

    | 🏷️ Model Name | 📜 Reference Paper | 📦 Source of Weights |
    |---------------|-------------------|---------------------|
    | Granite Speech | [Granite-speech: open-source speech-aware LLMs with strong English ASR capabilities](https://arxiv.org/abs/2505.08699) | `transformers` |
    | Granite Speech Plus | [ibm-granite/granite-speech-4.1-2b-plus](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus) | `transformers` |


## 📜 License

This project leverages [timm](https://github.com/huggingface/pytorch-image-models#licenses) and [transformers](https://github.com/huggingface/transformers#license) for converting pretrained weights from PyTorch to Keras. For licensing details, please refer to the respective repositories.

- 🔖 **kerasformers Code**: This repository is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).


## 🌟 Credits

- The [Keras](https://github.com/keras-team/keras) team for their powerful and user-friendly deep learning framework
- The [Transformers](https://github.com/huggingface/transformers) library for its robust tools for loading and adapting pretrained models
- The [pytorch-image-models (timm)](https://github.com/huggingface/pytorch-image-models) project for pioneering many computer vision model implementations
- All contributors to the original papers and architectures implemented in this library

## Citing

### BibTeX

```bash
@misc{gc2025kerasformers,
  author = {Gitesh Chawda},
  title = {KerasFormers},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/IMvision12/KerasFormers}}
```
