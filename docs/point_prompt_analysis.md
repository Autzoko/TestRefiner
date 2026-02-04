# Point Prompt Quality Analysis for UltraSAM

## Executive Summary

This analysis investigates how point prompt quality affects UltraSAM segmentation performance on the BUSI (Breast Ultrasound Images) dataset. We compare ground truth (GT) derived point prompts against TransUNet-generated point prompts and conduct a perturbation study to quantify sensitivity to point accuracy.

**Key Findings:**
- GT point prompts achieve **0.853 Dice** vs TransUNet's **0.806 Dice** (4.7% gap)
- **91.2%** of TransUNet points fall inside the lesion area
- When TransUNet points are inside the lesion, performance is nearly identical (0.6% difference)
- When points fall outside the lesion (8.8% of cases), performance drops catastrophically (Dice: 0.67 → 0.20)
- Point accuracy has a **strong correlation (r=0.74)** with segmentation quality

---

## Part I: TransUNet vs GT Point Prompt Comparison

### 1.1 Overall Results

| Metric | Value |
|--------|-------|
| Total Samples | 647 |
| GT Point Avg Dice | 0.8528 ± 0.15 |
| TransUNet Point Avg Dice | 0.8057 ± 0.24 |
| Average Difference | 0.0471 (GT better) |
| TransUNet Points in Lesion | 590 (91.2%) |
| TransUNet Points Outside Lesion | 57 (8.8%) |

### 1.2 Point Distance Statistics

| Statistic | Value |
|-----------|-------|
| Mean Distance | 27.0 px |
| Median Distance | 6.4 px |
| Std Deviation | 50.3 px |
| Min Distance | 0.0 px |
| Max Distance | 395.1 px |

The highly skewed distribution (mean >> median) indicates that most TransUNet predictions are accurate, but a subset of cases have severely mislocalized predictions.

### 1.3 Stratified Analysis: Point Location Matters

#### When TransUNet Point IS Inside Lesion (590 samples, 91.2%)

| Metric | GT Point | TransUNet Point | Difference |
|--------|----------|-----------------|------------|
| Avg Dice | 0.8701 | 0.8641 | 0.0060 |
| Avg Distance | - | 15.15 px | - |

**Insight:** When TransUNet correctly localizes within the lesion, UltraSAM achieves nearly identical performance regardless of which point is used. The 0.6% difference is negligible.

#### When TransUNet Point is OUTSIDE Lesion (57 samples, 8.8%)

| Metric | GT Point | TransUNet Point | Difference |
|--------|----------|-----------------|------------|
| Avg Dice | 0.6735 | 0.2019 | 0.4716 |
| Avg Distance | - | 149.71 px | - |

**Insight:** When TransUNet misses the lesion entirely, UltraSAM fails catastrophically. The 47% Dice drop demonstrates that point prompts must be inside the target region.

### 1.4 Performance by Distance Bucket

| Distance Range | Samples | Dice Difference | % In Lesion |
|----------------|---------|-----------------|-------------|
| 0-10 px | 386 | -0.0004 | 100% |
| 10-20 px | 80 | -0.0041 | 99% |
| 20-40 px | 68 | +0.0222 | 94% |
| 40-80 px | 47 | +0.0064 | 89% |
| 80-200 px | 55 | +0.3830 | 35% |

**Key Observation:** Performance degradation is minimal until distance exceeds ~40 pixels. Beyond 80 pixels, most points fall outside the lesion and performance collapses.

### 1.5 Cases Where TransUNet Outperforms GT

Surprisingly, in **2.5% of cases (16 samples)**, TransUNet points yielded better segmentation than GT centroid points. Analysis of these cases reveals:

| Case | Distance | GT Dice | TransUNet Dice | Difference |
|------|----------|---------|----------------|------------|
| benign_(100) | 55.3 px | 0.022 | 0.769 | -0.747 |
| benign_(304) | 16.2 px | 0.299 | 0.822 | -0.523 |
| malignant_(156) | 55.2 px | 0.440 | 0.792 | -0.352 |

**Explanation:** These cases likely have irregular lesion shapes where the geometric centroid falls in a suboptimal location (e.g., concave regions, thin protrusions). TransUNet's prediction, while not at the centroid, may land in a more "representative" area of the lesion.

---

## Part II: Perturbation Study

### 2.1 Methodology

Starting from GT centroid points, we added Gaussian noise with varying standard deviations (σ) and measured the impact on UltraSAM performance.

### 2.2 Results

| Perturbation σ | Actual Avg Distance | Avg Dice | Dice Drop |
|----------------|---------------------|----------|-----------|
| 0 (baseline) | 0.0 px | 0.8528 | - |
| 5 px | 6.1 px | 0.8524 | 0.0004 |
| 10 px | 12.4 px | 0.8510 | 0.0018 |
| 20 px | 24.8 px | 0.8375 | 0.0153 |
| 30 px | 38.8 px | 0.7969 | 0.0559 |
| 50 px | 62.2 px | 0.7164 | 0.1364 |

### 2.3 Observations

1. **High Tolerance at Small Perturbations:** UltraSAM is remarkably robust to small positional errors (σ ≤ 10 px), with less than 0.2% Dice degradation.

2. **Gradual Degradation:** Performance degrades smoothly rather than cliff-like, suggesting UltraSAM has learned spatial context.

3. **Critical Threshold:** Around σ = 30-50 px, degradation becomes significant (5-14% Dice drop), as perturbations increasingly push points outside lesion boundaries.

4. **Comparison with TransUNet:** TransUNet's average distance of 27 px corresponds to approximately σ = 20-30 px perturbation level, which aligns with its observed performance gap.

---

## Part III: Why TransUNet-Generated Points Underperform

### 3.1 Root Causes

#### 1. **Segmentation Errors Propagate to Point Generation**

TransUNet achieves ~75-80% Dice on BUSI, meaning 20-25% of predictions have significant errors. When TransUNet:
- **Under-segments:** The predicted centroid shifts toward the detected portion
- **Over-segments:** The centroid may move outside the true lesion
- **Completely misses:** The centroid lands on false positive regions

#### 2. **Centroid Sensitivity to Shape Errors**

The centroid is computed as the mean of all foreground pixels. This makes it highly sensitive to:
- **False positive blobs:** Pull centroid away from true lesion
- **Fragmented predictions:** Centroid may fall in background between fragments
- **Boundary inaccuracies:** Systematic over/under-estimation shifts centroid

#### 3. **Challenging Cases Compound Errors**

Analysis by GT detectability shows a clear pattern:

| GT Dice Range | Samples | Avg Distance | Avg Diff | % In Lesion |
|---------------|---------|--------------|----------|-------------|
| 0.85-1.0 | 460 | 15.8 px | 0.029 | 96% |
| 0.70-0.85 | 113 | 39.6 px | 0.107 | 86% |
| 0.50-0.70 | 43 | 64.7 px | 0.114 | 79% |
| 0.00-0.50 | 31 | 95.6 px | 0.011 | 58% |

Cases that are inherently difficult (low GT Dice) are also where TransUNet struggles most, creating a compounding effect.

#### 4. **Dataset-Specific Challenges**

BUSI contains:
- **Variable lesion sizes:** Small lesions have tighter tolerance for point accuracy
- **Low contrast boundaries:** Makes segmentation difficult
- **Benign vs malignant differences:** Different morphological characteristics affect both segmentation and point sensitivity

### 3.2 Failure Mode Analysis

Examining the worst 10 cases (Dice diff > 0.86):

| Case | Distance | GT Dice | TU Dice | In Lesion |
|------|----------|---------|---------|-----------|
| benign_(344) | 169.4 px | 0.945 | 0.000 | No |
| malignant_(173) | 227.2 px | 0.939 | 0.000 | No |
| benign_(153) | 47.8 px | 0.926 | 0.000 | No |
| malignant_(45) | 178.9 px | 0.909 | 0.029 | Yes |

**Pattern:** In catastrophic failures, TransUNet typically:
1. Detects a completely wrong region (false positive)
2. Fails to detect the lesion entirely (false negative with noise)
3. Provides severely fragmented predictions

---

## Part IV: Implications and Recommendations

### 4.1 For the UltraRefiner Pipeline

1. **Point Inside Lesion is Critical:** The most important factor is whether the point falls inside the lesion. Exact position within the lesion matters little.

2. **TransUNet is Sufficient for Most Cases:** 91% of cases work well. Focus optimization on the failing 9%.

3. **Consider Confidence-Based Filtering:** Reject TransUNet predictions with low confidence to avoid catastrophic failures.

### 4.2 For Improving Point Prompt Quality

1. **Multiple Point Sampling:** Instead of centroid, sample multiple points and use UltraSAM's multi-point prompt capability.

2. **Skeleton-Based Points:** Use morphological skeleton to find points that are maximally inside the shape.

3. **Ensemble Predictions:** Combine multiple TransUNet models to reduce false positives.

4. **Iterative Refinement:** Use UltraSAM's output to refine point selection in a feedback loop.

### 4.3 For Model Training

1. **Data Augmentation:** Train with perturbed point prompts (σ = 10-30 px) to improve robustness.

2. **Hard Example Mining:** Focus training on cases where TransUNet fails.

3. **Joint Training:** Consider end-to-end training of detector + segmentor.

---

## Conclusions

1. **UltraSAM is robust to moderate point inaccuracies** (~10-20 px), but fails when points land outside the target region.

2. **TransUNet's 91.2% in-lesion rate is good but not sufficient** for clinical deployment where catastrophic failures in 8.8% of cases are unacceptable.

3. **The performance gap (4.7% Dice) is almost entirely explained by out-of-lesion points.** When TransUNet points are correct, performance is equivalent to GT points.

4. **Future work should focus on:**
   - Detecting unreliable TransUNet predictions
   - Improving robustness to out-of-distribution points
   - Developing better point selection strategies beyond centroid

---

## Appendix: Experimental Details

- **Dataset:** BUSI (Breast Ultrasound Images), 647 samples across 5 cross-validation folds
- **TransUNet:** R50-ViT-B_16, trained with 5-fold cross-validation
- **UltraSAM:** Pre-trained weights, point prompt mode
- **Point Generation:** Centroid of binary mask
- **Perturbation:** Gaussian noise with σ ∈ {5, 10, 20, 30, 50} pixels
- **Metrics:** Dice coefficient, point distance (Euclidean)

---

*Analysis generated from experimental results in `outputs/analysis/point_prompts/busi/`*
