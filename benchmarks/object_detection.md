# Object Detection — Per-Variant COCO AP & Params

COCO val2017 box AP and parameter counts for **every object detection variant in kmodels**, taken from the **original publication's main results table** for that variant. `Params (M)` is computed by directly instantiating the model from the kmodels registry. Variants are listed in increasing parameter order within each family.

The metric is COCO val2017 single-scale **box AP** (not multi-scale TTA). Each family section cites the exact paper Table the values came from in its intro paragraph. `—` means the paper doesn't report that variant or the value is not reliably documented.

All variants below are loaded with `<Family>Detect.from_weights("<variant>")`.

---

### DETR &mdash; [paper](https://arxiv.org/abs/2005.12872)


| Variant | Box AP | AP50 | AP75 | Params (M) |
|---------|------:|----:|----:|-----------:|
| `detr-resnet-50`  | 42.0 | 62.4 | 44.2 | 41 |
| `detr-resnet-101` | 43.5 | 63.8 | 46.4 | 60 |

### RT-DETR &mdash; [paper](https://arxiv.org/abs/2304.08069)


| Variant | Box AP | Params (M) |
|---------|------:|-----------:|
| `rtdetr-r18vd`            | 46.5 | 20 |
| `rtdetr-r18vd-coco-o365`  | 49.2 | 20 |
| `rtdetr-r34vd`            | 48.9 | 31 |
| `rtdetr-r50vd`            | 53.1 | 43 |
| `rtdetr-r50vd-coco-o365`  | 55.3 | 43 |
| `rtdetr-r101vd`           | 54.3 | 76 |
| `rtdetr-r101vd-coco-o365` | 56.2 | 76 |

### RT-DETRv2 &mdash; [paper](https://arxiv.org/abs/2407.17140)


| Variant | Box AP | Params (M) |
|---------|------:|-----------:|
| `rtdetr-v2-r18vd`  | 47.9 | 20 |
| `rtdetr-v2-r34vd`  | 49.9 | 31 |
| `rtdetr-v2-r50vd`  | 53.4 | 43 |
| `rtdetr-v2-r101vd` | 54.3 | 76 |

### D-FINE &mdash; [paper](https://arxiv.org/abs/2410.13842)


| Variant | Weights | Box AP | Params (M) |
|---------|---------|------:|-----------:|
| `dfine-nano`   | COCO      | 42.8 | 3.8 |
| `dfine-nano`   | COCO+O365 | 44.2 | 3.8 |
| `dfine-small`  | COCO      | 48.7 | 10  |
| `dfine-small`  | COCO+O365 | 50.7 | 10  |
| `dfine-medium` | COCO      | 52.3 | 19  |
| `dfine-medium` | COCO+O365 | 55.1 | 19  |
| `dfine-large`  | COCO      | 54.0 | 31  |
| `dfine-large`  | COCO+O365 | 57.1 | 31  |
| `dfine-xlarge` | COCO      | 55.8 | 63  |
| `dfine-xlarge` | COCO+O365 | 59.3 | 63  |

### RF-DETR &mdash; [paper](https://arxiv.org/abs/2511.09554)


| Variant | Box AP | AP50 | AP75 | Params (M) |
|---------|------:|----:|----:|-----------:|
| `rfdetr-nano`   | 48.0 | 67.0 | 51.4 | 27 |
| `rfdetr-small`  | 52.9 | 71.9 | 57.0 | 28 |
| `rfdetr-medium` | 54.7 | 73.5 | 59.2 | 30 |
| `rfdetr-base`   | —    | —    | —    | 27 |
| `rfdetr-large`  | 56.5 | 75.1 | 61.3 | 30 |
