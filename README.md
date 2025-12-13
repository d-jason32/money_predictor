Dataset from https://www.kaggle.com/datasets/aishwaryatechie/usd-bill-classification-dataset. This project trains a MobileNetV3-based classifier to recognize USD bill denominations from images and provides a simple CLI for inference against the bundled dataset or your own photos.

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
## Inference 50 Dollar Image
![img_1.png](img_1.png)
## Result
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