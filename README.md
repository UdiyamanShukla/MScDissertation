# MSc Dissertation: A Comparison of Segmentation- and Classification-based Approaches for Blink Detection



The project presents a comparative analysis of modern deep learning methods for blink detection, contrasting **classification-based Convolutional Neural Networks (CNNs)** with a **explainable segmentation-based approach** using Meta's Segment Anything Model 2 (SAM2).

## Abstract
Blink detection is a core computer vision task with applications in medical diagnostics, driver monitoring, and human-computer interaction. This project explores using the Segment Anything Model (SAM2) to create a transparent pipeline that generates segmentation masks of the eye region, allowing for heuristic-based classification. The segmentation approach achieved an accuracy of **94.1%** and provides key advantages in explainability, such as identifying mislabelled data.

**Key Findings:**
-   **Classification (CNNs):** High accuracy (**InceptionResNetV2: 98.1%**, MobileNetV2: 97.3%).
-   **Segmentation (SAM2 + Heuristics):** High explainability, good performance (**94.1%**), and the ability to audit and clean dataset labels.

## Repository Structure & Code Files

### 1. Classification-Based Approaches (CNN)
These scripts implement transfer learning for CNN classifiers.

| File | Description |
| :--- | :--- |
| `InceptionV2.py` | Python script for training and fine-tuning the **InceptionResNetV2** model on the blink detection task. |
| `MobileNet_Implementation_fullrun.ipynb` | Jupyter Notebook for training and fine-tuning the **MobileNetV2** model. |

### 2. Segmentation-Based Approach (SAM2 - Primary Method)
This is the core contribution of the project, using SAM2 and heuristic rules for blink detection.

| File | Description |
| :--- | :--- |
| `Aspect_ratioed_eccentricity_SAM2_contour_analysis_compiler_model_tester.ipynb` | **Main Testing Script.** Uses the finalized heuristic rules to classify images as 'blink' or 'open' using SAM2-generated masks. |
| `Revamped_Error_analysis_SAM2_compiler.ipynb` | **Model Development Script.** Used for exploratory analysis, developing heuristics (mask area, score, eccentricity), and error analysis. |
| `Ellipse_eccentricity_aspect_ratio.ipynb` | **Supporting Analysis Script.** Contains code for calculating key geometric properties (eccentricity, aspect ratio) from segmentation masks. |

### 3. Alternative Segmentation Models (Exploratory)
Initial experiments with other segmentation architectures.

| File | Description |
| :--- | :--- |
| `Deeplab3_diagnose_issues_analyzer.ipynb` | Exploratory implementation and analysis of **DeepLabV3+** for eye region segmentation. |
| `U-Net_Diagnose_issues_analyzer.ipynb` | Exploratory implementation and analysis of **U-Net** for eye region segmentation. |

### 4. Data
| File | Description |
| :--- | :--- |
| `consolidated_coordinates.csv` | Processed CSV file containing AI-predicted eye center coordinates for images used in the segmentation pipeline. |


### Prerequisites

*   **Python 3.9**
*   Key Libraries: `torch`, `torchvision`, `tensorflow`, `keras`, `opencv-python`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`
*   For SAM2: Install `segment-anything` package and download the `sam2.1_hiera_large.pt` model checkpoint from Meta's official repository (https://github.com/facebookresearch/segment-anything)


### Usage

1.  **Data Preparation:** Organize your image dataset into a flat directory structure per participant. Ensure the `consolidated_coordinates.csv` file is in the correct path.
2.  **Run Classification Models:** Execute the `InceptionV2.py` script or the `MobileNet_Implementation_fullrun.ipynb` notebook to train and evaluate the CNN models.
3.  **Run Segmentation Pipeline:** The main workflow is in `Aspect_ratioed_Revamped_on_new_CSV_SAM2_contour_analysis_compiler_Tester1.ipynb`. This notebook loads images, prompts SAM2 with eye coordinates, applies heuristic rules, and performs classification.

## Methodology Overview

1.  **Data Preprocessing:** Images were resized and organized. Eye center coordinates were filtered and consolidated.
2.  **Classification (CNN):** Implemented transfer learning with a two-phase strategy: (i) training only the final layer, (ii) fine-tuning all layers at a low learning rate.
3.  **Segmentation (SAM2):**
    *   **Prompting:** SAM2 was prompted with eye center coordinates to generate candidate segmentation masks.
    *   **Heuristic Development:** Rules based on mask area, score, eccentricity, and globule size ratio were developed to override SAM2's default mask selection.
    *   **Classification:** The final selected mask was classified based on its pixel area.
4.  **Evaluation:** Models were evaluated on a held-out validation set using accuracy, precision, recall, and F1-score.


## Future Work

Potential improvements and next steps, as outlined in the thesis, include:
-   Integrating pupil detection heuristics to handle highly eccentric eye shapes.
-   Generalizing heuristic rules to be resolution-invariant.
-   Fine-tuning U-Net or DeepLabV3+ with a dedicated eye segmentation dataset.
-   Experimenting with smaller, faster variants of SAM2 for real-time applications.

## Author

**Udiyaman Shukla**
-   University of Bristol, MSc in Data Science

## Acknowledgments
-   Supervisor: Dr. Gabriella Miles, University of Bristol.
-   The dataset was provided via the Open Science Framework (OSF).
