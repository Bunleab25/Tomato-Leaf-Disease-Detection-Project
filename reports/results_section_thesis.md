# Chapter 4: Results and Discussion

## 4.1 Chapter Overview
This chapter presents the comparative performance of a custom convolutional neural network (CNN) baseline and a transfer-learning MobileNetV2 model for tomato leaf disease classification. The chapter is organized as follows: Section 4.2 describes the experimental protocol, Section 4.3 reports aggregate quantitative outcomes, Section 4.4 discusses class-level behavior, Section 4.5 interprets findings in context, and Sections 4.6 and 4.7 conclude with implications and future research directions.

## 4.2 Experimental Setup and Evaluation Protocol
The experiments used the 5-class tomato leaf dataset in `dataset_5class` (Blight, Healthy, Leaf_Mold, Septoria, and Virus). A deterministic 70/15/15 split (train/validation/test) was used with random seed 42 to ensure repeatability. Both models were trained and evaluated under the same split policy, and final performance was measured on the held-out test set.

Having established the protocol, the next section presents the overall quantitative comparison between the two models.

## 4.3 Quantitative Performance Comparison
Table 4.1 summarizes the overall test-set performance.

| Model | Test Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Custom CNN | 0.8000 | 0.7885 | 0.7938 |
| MobileNetV2 | 0.9800 | 0.9796 | 0.9800 |

MobileNetV2 improved test accuracy by 0.1800 (18.00 percentage points) relative to the custom CNN baseline. Improvements are also visible in Macro F1 and Weighted F1, indicating stronger and more balanced performance across classes.

To further understand where these gains originate, the following section examines per-class results.

## 4.4 Class-Level Results and Error Behavior
Table 4.2 reports per-class F1 scores.

| Class | CNN F1 | MobileNetV2 F1 |
|---|---:|---:|
| Blight | 0.7415 | 0.9593 |
| Healthy | 0.8737 | 0.9947 |
| Leaf_Mold | 0.8516 | 0.9764 |
| Septoria | 0.6119 | 0.9760 |
| Virus | 0.8638 | 0.9917 |

The most significant class-level gain appears in Septoria, where MobileNetV2 markedly improved discrimination compared with the CNN baseline. Under the improved setup, all MobileNetV2 per-class F1 scores are above 0.95.

Given these class-level outcomes, the next section discusses their practical and methodological implications.

## 4.5 Discussion and Interpretation
The results suggest that transfer learning with MobileNetV2 provides more effective feature representations than the custom CNN under the current dataset size and variability. This advantage is reflected not only in higher overall accuracy, but also in improved macro-level balance across classes.

From an application standpoint, MobileNetV2 is better suited as the primary model for subsequent optimization and potential deployment. The custom CNN remains valuable as a baseline for methodological benchmarking and ablation-style comparisons.

Building on this interpretation, the chapter now states the primary conclusion.

## 4.6 Conclusion
Under identical training and evaluation conditions, MobileNetV2 consistently outperformed the custom CNN baseline. Therefore, MobileNetV2 is selected as the primary model for the next development stage in this project.

Although performance is strong, several constraints remain and are addressed below.

## 4.7 Limitations and Future Work
Despite the favorable outcomes, this study has several limitations. First, evaluation was conducted on one deterministic split, which may not fully represent data-partition variance. Second, residual confusion remains between visually similar disease patterns despite overall gains. Third, the current analysis emphasizes predictive quality and does not yet report deployment metrics such as latency, throughput, and memory cost on constrained hardware.

Future work should prioritize the following:
1. Perform k-fold cross-validation to provide more robust and stable performance estimates.
2. Apply staged fine-tuning by unfreezing deeper MobileNetV2 layers after classifier warm-up.
3. Evaluate imbalance-aware objectives (for example weighted cross-entropy or focal loss) to improve difficult-class recall.
4. Benchmark additional lightweight backbones (for example EfficientNet-B0 and MobileNetV3) under the same protocol.
5. Add deployment-oriented evaluation (latency, memory footprint, throughput) for production-aware model selection.
