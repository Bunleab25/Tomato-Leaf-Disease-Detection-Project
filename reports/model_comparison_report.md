# Tomato Leaf Disease Classification: CNN vs MobileNetV2

## Objective
Compare a custom CNN baseline with MobileNetV2 on the same tomato leaf 5-class dataset using the same deterministic 70/15/15 split.

## Experimental Setup
- Dataset: `dataset_5class`
- Classes: Blight, Healthy, Leaf_Mold, Septoria, Virus
- Split: 70% train / 15% validation / 15% test
- Random seed: 42
- Device: CUDA
- Evaluation script: `scripts/evaluation/evaluate_models.py`

## Final Test Results
| Metric | CNN (custom) | MobileNetV2 |
|---|---:|---:|
| Accuracy | 0.8000 | 0.9800 |
| Macro F1 | 0.7885 | 0.9796 |
| Weighted F1 | 0.7938 | 0.9800 |

Absolute accuracy gain (MobileNetV2 over CNN): **+0.1800** (18.00 percentage points).

## Per-Class F1 Score
| Class | CNN | MobileNetV2 |
|---|---:|---:|
| Blight | 0.7415 | 0.9593 |
| Healthy | 0.8737 | 0.9947 |
| Leaf_Mold | 0.8516 | 0.9764 |
| Septoria | 0.6119 | 0.9760 |
| Virus | 0.8638 | 0.9917 |

## Key Findings
1. MobileNetV2 outperformed the custom CNN on all aggregate metrics.
2. The largest class-level improvement was on Septoria (F1 improved from 0.6119 to 0.9760).
3. All MobileNetV2 class-wise F1 scores are above 0.95 in the improved training configuration.

## Conclusion
MobileNetV2 is substantially stronger for this dataset under the improved training setup and should be selected as the primary model for report conclusions and next-stage optimization.

## Reproducibility Commands
```bash
python scripts/evaluation/evaluate_models.py --model cnn --weights /path/to/cnn_model.pth
python scripts/evaluation/evaluate_models.py --model mobilenet --weights mobile/mobilenet_model_best.pth
```
