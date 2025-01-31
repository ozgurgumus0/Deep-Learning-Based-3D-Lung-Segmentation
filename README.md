# Deep-Learning Based 3D Lung Segmentation

## Project Overview  
This project presents a **deep-learning-based 3D lung segmentation model** designed to segment lung regions in **Computed Tomography (CT) scans**. The model is built using a **U-Net architecture** and trained on a dataset derived from the **AAPM Thoracic Auto-Segmentation Challenge**. It incorporates **preprocessing techniques, advanced data augmentation, and optimized hyperparameters** to enhance segmentation accuracy while mitigating overfitting.  

## Features  
- **3D U-Net architecture** optimized for **medical image segmentation**  
- **Preprocessing techniques** (HU windowing, Gaussian smoothing, standardization)  
- **Advanced data augmentation** (elastic deformation, rotations, flips, brightness variations)  
- **Dynamic learning rate adjustment** using ReduceLROnPlateau  
- **Evaluation metrics:** **Dice Similarity Coefficient (DSC), Intersection over Union (IoU)**  
- **GPU-accelerated training** using TensorFlow/Keras  

## Dataset  
The dataset consists of **3D volumetric CT scans** in **DICOM/NIfTI formats**. Preprocessing ensures **uniformity and consistency** in medical images for better segmentation performance.  

### Preprocessing Steps:  
- **HU Windowing:** Intensity values clipped to [-1000, 400] HU to focus on lung tissues.  
- **Gaussian Smoothing:** Reduces noise and artifacts for improved segmentation.  
- **Standardization:** Mean subtraction and division by standard deviation to normalize pixel values.  
- **Normalization:** Ensures consistency across different medical imaging conditions.  

## Model Architecture  
The **3D U-Net** model consists of:  
- **Encoder (Contracting Path):** Captures hierarchical features using **3D convolutional layers** with **ReLU activation**.  
- **Decoder (Expanding Path):** Upsamples and reconstructs segmentation maps with **skip connections**.  
- **Skip Connections:** Preserve spatial details from encoder layers, improving segmentation accuracy.  
- **Dropout & Batch Normalization:** Prevent overfitting and enhance generalization.  

## Training  
- **Optimizer:** Adam (β1=0.9, β2=0.999)  
- **Loss Function:** Dice Loss + Binary Cross-Entropy  
- **Learning Rate:** 1e-3 (adaptive decay using ReduceLROnPlateau)  
- **Batch Size:** 4  
- **Epochs:** 100 (early stopping at epoch 97)  
- **GPU:** NVIDIA Tesla P100  

## Evaluation & Results  
The model was tested on unseen data using **DSC, IoU, and binary cross-entropy loss**.  

**Results:**  
- **Test Dice Score:** **0.9432** (high segmentation accuracy)  
- **Test IoU:** **0.8926** (strong spatial agreement with ground truth)  
- **Test Loss:** **0.0440** (low error rate in segmentation predictions)  

### Visualizations  
- **Predicted segmentation masks overlayed on CT scans** for qualitative assessment.  

## Installation & Setup  

### Requirements  
- Python 3.8+  
- TensorFlow 2.x  
- Keras  
- NumPy  
- Matplotlib  
- SimpleITK  
- OpenCV  
- Pydicom  
- Scikit-image  

### Installation  
```bash
pip install tensorflow keras numpy matplotlib simpleitk opencv-python pydicom scikit-image
```

## Run the Model
- **Preprocess Data** 
```bash
python lung-segmentation.ipynb
```

## Future Work

- Integration of attention mechanisms into U-Net for better feature extraction.
- Further dataset expansion to improve generalization on diverse lung conditions.
- Implementation of hybrid CNN-Transformer architectures for enhanced 3D segmentation.

## Code & Model Repository
🔗 [Project Repository](https://github.com/keremerciyes/Deep-Learning-Based-3D-Lung-Segmentation)