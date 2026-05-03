# Robust Chest X-ray Classification

This repository presents a deep learning project for **binary chest X-ray classification** on the **RSNA Pneumonia Detection Challenge** dataset. The work is organized into two notebook-based implementations and a written report:

- `code1(overfitting_problem).ipynb`
- `code2(overcoming_overfitting_problem).ipynb`
- `Report.docx`

The project uses a **transfer learning** approach with **ResNet-18** to classify chest radiographs as **Normal** or **Disease / Pneumonia**.

---

## Project Objective

The primary objective of the project is to build a robust classification pipeline that can distinguish between normal and pneumonia-affected chest X-ray images while monitoring and reducing overfitting.

The work is centered on two stages:

1. **Initial baseline training** that demonstrates the overfitting problem.
2. **A shorter training regime** that reduces overfitting and improves the training–validation balance.

---

## Dataset

The project is based on the **RSNA Pneumonia Detection Challenge** dataset. The notebooks work with:

- `stage_2_train_labels.csv`
- `stage_2_detailed_class_info.csv`
- `stage_2_train_images`
- `stage_2_test_images`

The code reads DICOM radiographs, converts them into RGB images, and resizes them to **224 × 224** for input into the network.

---

## Workflow Summary

### 1. Data preparation
- Labels are loaded from the RSNA CSV files.
- Chest X-ray DICOM images are read with `pydicom`.
- Images are converted to uint8 format and then transformed into RGB.
- A standard image preprocessing pipeline is applied:
  - `RandomHorizontalFlip`
  - `Resize(224)`
  - `ToTensor()`

### 2. Model construction
- A pretrained **ResNet-18** backbone is used.
- The original classifier head is replaced with a **2-class linear layer**.
- Training is performed with **SGD** and a **StepLR** scheduler.
- The loss function is **CrossEntropyLoss**.

### 3. Evaluation and interpretation
- Validation accuracy is computed after training.
- Saliency maps are generated to visualize which regions influence predictions.
- Confusion matrices are plotted for class-wise performance inspection.

---

## Notebook-by-Notebook Analysis

### `code1(overfitting_problem).ipynb`
This notebook implements the initial version of the pipeline and trains the ResNet-18 model for **15 epochs**.

#### What happens in this notebook
- The RSNA data is loaded and split into training and validation sets.
- A pretrained ResNet-18 model is fine-tuned for binary classification.
- Training is performed for a relatively long duration.
- Saliency maps are produced for selected samples.
- A validation confusion matrix is generated.

#### Observed output behavior
- Training accuracy rises very high by the end of training.
- Final logged training accuracy reaches approximately **98.96%**.
- Final logged validation accuracy reaches approximately **85.51%**.
- The gap between training and validation performance indicates **overfitting**.

#### Confusion matrix insight
The validation confusion matrix shows that the model performs reasonably well, but still makes misclassifications in both classes, which aligns with the overfitting observed during training.

---

### `code2(overcoming_overfitting_problem).ipynb`
This notebook keeps the same overall architecture and preprocessing design, but reduces the training duration to **8 epochs**.

#### What happens in this notebook
- The same RSNA preprocessing and ResNet-18 transfer learning pipeline is reused.
- The model is trained for fewer epochs to control memorization.
- Saliency maps are again produced for selected images.
- A confusion matrix is created for a broader prediction set.

#### Observed output behavior
- Training accuracy remains lower than in the first notebook.
- Final logged training accuracy reaches approximately **91.84%**.
- Final logged validation accuracy reaches approximately **82.83%**.
- The training–validation gap is smaller, which suggests a more restrained fit and less overfitting pressure.

#### Important note
The report presents this notebook as the overfitting-mitigation stage. From the recorded outputs, the main benefit is not a higher final validation score, but a **more controlled training process** and a **smaller overfitting gap**.

---

## Key Results

| Notebook | Epochs | Final Training Accuracy | Final Validation Accuracy | Main Interpretation |
|---|---:|---:|---:|---|
| `code1` | 15 | ~98.96% | ~85.51% | Strong fit, but clear overfitting |
| `code2` | 8 | ~91.84% | ~82.83% | Reduced memorization and more controlled training |

---

## Outputs Produced by the Project

The notebooks generate the following outputs:

- Dataset distribution summaries
- Sample chest X-ray visualizations
- Training and validation progress logs
- Saliency maps for interpretability
- Confusion matrices
- Saved model weights:
  - `Transfer_Learning_Resnet.pt`

---

## Repository Structure

```text
.
├── code1(overfitting_problem).ipynb
├── code2(overcoming_overfitting_problem).ipynb
├── Report.docx
├── README.md
├── LICENSE
└── requirements.txt
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Khubaib-Muhammad/Robust-Chest-X-ray-Classification-rsna-pneumonia-detection-challenge-
cd Robust-Chest-X-ray-Classification-rsna-pneumonia-detection-challenge-
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Open a notebook
Launch Jupyter Notebook or JupyterLab and run either:
- `code1(overfitting_problem).ipynb`
- `code2(overcoming_overfitting_problem).ipynb`

---

## Dependencies

The project relies on the following key libraries:

- PyTorch
- TorchVision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Pillow
- pydicom
- scikit-learn
- tqdm
- kagglehub

Additional utilities used in the notebooks include:

- albumentations
- timm
- torchmetrics
- netcal

---

## Interpretation of the Project

This repository demonstrates an important machine learning lesson in medical imaging: a model can achieve very high training accuracy and still generalize poorly if it is allowed to overfit.

The first notebook shows the overfitting issue clearly. The second notebook addresses that issue by reducing training time, resulting in a more disciplined fitting process. Together, the two notebooks provide a complete comparison between an overfit baseline and a more controlled training strategy.

---

## License

This project is released under the MIT License. See the `LICENSE` file for details.

---

## Author

**Khubaib Muhammad** :https://github.com/Khubaib-Muhammad
**Kashan Maqsood** : https://github.com/KashanMaqsood
