# CIFAR‑10 Image Classification using Deep Learning

## Project Summary
This project implements Convolutional Neural Networks (CNNs) to classify images from the CIFAR‑10 dataset. A baseline model was first developed and then improved using modern deep learning techniques to increase performance and generalization.

---

## Dataset
The CIFAR‑10 dataset contains:
- 60,000 RGB images
- Image size: 32 × 32 pixels
- 10 object categories:
  - Airplane, Automobile, Bird, Cat, Deer  
  - Dog, Frog, Horse, Ship, Truck

Dataset split:
- Training images: 50,000  
- Test images: 10,000  

The official dataset split was used for training and evaluation.

---

## Data Preprocessing
The following preprocessing techniques were applied:

- Pixel normalization (scaled from 0–255 to 0–1)
- One‑hot encoding of labels
- Data augmentation on training data:
  - Random rotations
  - Horizontal flips
  - Width and height shifting
  - Random zoom

These steps improve model robustness and reduce overfitting.

---

## Models Implemented

### Baseline CNN
Architecture:
- Two convolutional layers
- Max‑pooling layers
- Fully connected classifier

**Test Accuracy:** 70.21%

#### Baseline Training Accuracy & Loss
*(Insert graph image here)*  
<img width="700" height="470" alt="download" src="https://github.com/user-attachments/assets/bf9e7424-3bcd-4547-a7ec-f56dba8471d5" />


#### Baseline Loss Curve
*(Insert graph image here)*  
<img width="691" height="470" alt="download" src="https://github.com/user-attachments/assets/b05e7647-3dbf-4473-b4b4-222e27799d92" />


---

### Improved CNN
Enhancements applied:
- Additional convolution layer
- Batch Normalization
- Dropout regularization (0.3)
- Data augmentation
- Early stopping and learning rate scheduling

**Test Accuracy:** 83.42%

#### Improved Model Accuracy & Loss
*(Insert graph image here)*  


---

## Performance Comparison

| Model | Test Accuracy |
|------|---------------|
| Baseline CNN | 70.58% |
| Improved CNN | 83.92% |

Overall improvement: **+13.7%**

---

## Confusion Matrix
The confusion matrix illustrates class‑wise prediction performance.

*(Insert confusion matrix image here)*  


---

## Key Observations
- Data augmentation significantly improved model generalization.
- Batch normalization stabilized the training process.
- Dropout reduced overfitting.
- Common misclassifications occurred between:
  - Cats and dogs
  - Trucks and automobiles
- Visually distinct classes such as airplanes and ships achieved higher accuracy.

---

## How to Run the Project

Train the model:
python train.py


Evaluate the model:
python evaluate.py


---

## Conclusion
This project demonstrates the effectiveness of CNNs for image classification. By improving the architecture and applying regularization and augmentation techniques, the final model achieved strong performance on the CIFAR‑10 dataset.
