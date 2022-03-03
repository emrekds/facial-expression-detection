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

##Model Performance

My cnn model performed %64 accuracy on private test data.

![images/accuracy.png]
