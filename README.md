# Facial Expression Detection

Detect facial expressions with pretrained convolutional neural network model or build your own. The model can predict 7 different expression based on data.
- Angry
- Disgust
- Fear
- Happy
- Sad
- Suprise
- Neutral


## Data

The dataset used here is `fer2013` that contains over 60000 facial expressions from kaggle's [FER challenge of 2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge).<br>
Data contains mostly happy expression.

### Download from Kaggle

- Download from [kaggle/fer2013](https://www.kaggle.com/deadskull7/fer2013?select=fer2013.csv)<br>

### Dowload with Kaggle API

- Install Kaggle from [github](https://github.com/Kaggle/kaggle-api)   
- Use the command in terminal `kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge`    

Docs on Kaggle API usage :
[github](https://github.com/Kaggle/kaggle-api) | [kaggle](https://www.kaggle.com/docs/api)

## Requirements

```python
tensorflow==2.6.0
matplotlib==3.5.1
pandas==1.4.1
seaborn==0.11.2
opencv-python==4.5.5
validators==0.18.2
numpy==1.21.5
```

## Model Performance

CNN model performed %64 accuracy on private test data.

![](images/accuracy.png)

## Usage

- Clone repository.

- Import or open fer.py
```python
import fer
```

- Create a class object.
```python
FER=fer.FaceExpressionRecognition()
```

- Use predict function.
```python
img,preds=FER.predict('images/happy1.png')
print(preds)
[{'Angry': 0.0004581761, 
  'Disgust': 7.457644e-09, 
  'Fear': 0.0014978538, 
  'Happy': 99.86222, 
  'Sad': 0.0007480914, 
  'Surprise': 0.024714392, 
  'Neutral': 0.110366486}]
plt.imshow(img,cmap='gray')
```
![](images/happy1.png)
