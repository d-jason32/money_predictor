Dataset from https://www.kaggle.com/datasets/aishwaryatechie/usd-bill-classification-dataset. 

## Introduction
Lightweight image classification project that recognizes USD bill denominations. The repo already includes the dataset and a trained model, so you can run inference or continue training without additional downloads.

## How it works
We fine-tune MobileNetV3-Large on the provided USD bill dataset with standard augmentations, save the best checkpoint and label map, and offer a CLI (`predict.py`) to classify new images using the trained weights.

## Tech Stack
- Python 3.10, PyTorch, Torchvision, TorchTriton
- MobileNetV3-Large backbone
- TQDM for progress, Pillow for image IO, Kaggle dataset (bundled locally)

## 1 Dollar Image
![img.png](img.png)
## Inference Result 
```
Predicted amount: $1 bill (Dollar)  [class='1 Dollar'] (p=0.9992)

Top candidates:
  $1 bill (Dollar)  [class='1 Dollar']: 0.9992
  $50 bill (Dollar)  [class='50 Dollar']: 0.0004
  $2 bill (Doolar)  [class='2 Doolar']: 0.0003
```
## 50 Dollar Image
![img_1.png](img_1.png)
## Inference Result
```
Predicted amount: $50 bill (Dollar)  [class='50 Dollar'] (p=1.0000)

Top candidates:
  $50 bill (Dollar)  [class='50 Dollar']: 1.0000
  $5 bill (Dollar)  [class='5 Dollar']: 0.0000
  $2 bill (Doolar)  [class='2 Doolar']: 0.0000
```
## 100 Dollar Image
![img_2.png](img_2.png)
## Inference Result
```
Predicted amount: $100 bill (Dollar)  [class='100 Dollar'] (p=1.0000)

Top candidates:
  $100 bill (Dollar)  [class='100 Dollar']: 1.0000
  $2 bill (Doolar)  [class='2 Doolar']: 0.0000
  $50 bill (Dollar)  [class='50 Dollar']: 0.0000```