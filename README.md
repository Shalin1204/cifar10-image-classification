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

<img width="700" height="470" alt="download" src="https://github.com/user-attachments/assets/bf9e7424-3bcd-4547-a7ec-f56dba8471d5" />


#### Baseline Loss Curve
  
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
 
![47477964-1e22-41c5-9301-2cd075e26ceb](https://github.com/user-attachments/assets/e859e530-81e2-488f-a9a2-1282471682e8)

![59f72a7a-3817-45d4-b8ed-0496894d5417](https://github.com/user-attachments/assets/6cd694b9-20b5-48c3-aaa8-f705d38be81c)



---

## Performance Comparison

| Model | Test Accuracy |
|------|---------------|
| Baseline CNN | 70.58% |
| Improved CNN | 83.92% |

Overall improvement: **+13.3%**

---

## Confusion Matrix
The confusion matrix illustrates class‑wise prediction performance.


![6e634fde-4679-4cef-9d3d-ccff4690cb69](https://github.com/user-attachments/assets/7cc1b1ed-5c6e-4148-bffe-17b9c3af7d8a)

 
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
