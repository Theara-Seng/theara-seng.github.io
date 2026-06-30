---
title: "Fruit Multiclass Classification using PyTorch and ResNet18: A Complete Image Classification Project"
date: 2026-01-06
permalink: /posts/2026-01-06-fruit-multiclass-classification-pytorch-resnet18
tags:

    - PyTorch
    - Deep Learning
    - CNN
    - ResNet18
    - Transfer Learning
    - Computer Vision
    - Image Classification
    - Multiclass Classification
    - Python
    - Torchvision

---

# Fruit Multiclass Classification using PyTorch and ResNet18: A Complete Image Classification Project

## 1. Introduction

In the previous PyTorch projects, I worked with binary classification and multiclass classification.

In the binary classification project, the model predicted one of two possible classes.

For example:

```text
chihuahua
muffin
```

That project was binary classification because the output had only two classes.

In this project, I continue with image classification, but the problem is now **multiclass classification**.

The model predicts one fruit class from several possible fruit classes.

The possible classes are:

```text
Apple
Banana
Grape
Mango
Strawberry
```

This project also introduces an important deep learning technique:

```text
Transfer Learning
```

Instead of training a CNN completely from zero, I use a pretrained model called:

```text
ResNet18
```

ResNet18 has already learned general image features from a large image dataset. Then I replace the final layer so that it can classify fruit images.

This article explains the complete workflow:

```text
download dataset
load image folders
prepare image transformations
build a pretrained ResNet18 model
train the model
use early stopping
save the best model
evaluate the model
read the confusion matrix
predict a new image
test with laptop camera
```

---

## 2. Project Goal

The main goal of this project is:

```text
Given one fruit image,
predict the correct fruit class.
```

Example:

```text
Input image: banana image
Output: Banana
```

The model predicts one of these five classes:

| Class Index | Class Name |
| ----------: | ---------- |
| 0 | Apple |
| 1 | Banana |
| 2 | Grape |
| 3 | Mango |
| 4 | Strawberry |

This is a **multiclass classification** problem because each image belongs to exactly one class.

In simple words:

```text
one image → one fruit class
```

---

## 3. Multiclass Classification

Multiclass classification means there are more than two possible classes, but only one class is correct for each image.

For this project:

```text
Apple OR Banana OR Grape OR Mango OR Strawberry
```

Only one answer should be selected.

For example, if the input image is a banana, the correct label is:

```text
Banana
```

The model should not predict Apple, Grape, Mango, or Strawberry for that image.

In PyTorch, multiclass classification usually uses:

```python
nn.CrossEntropyLoss()
```

During prediction, the model output is converted into probabilities using:

```python
torch.softmax(outputs, dim=1)
```

Then the highest probability is selected using:

```python
torch.argmax(outputs, dim=1)
```

---

## 4. Multiclass vs Multi-label Classification

It is important to understand the difference between multiclass classification and multi-label classification.

### Multiclass Classification

```text
one image → one class
```

Example:

```text
image1.jpg → Banana
```

The label is one number:

```text
1
```

### Multi-label Classification

```text
one image → many possible classes
```

Example:

```text
image1.jpg → Apple + Banana + Grape
```

The label is a vector:

```text
[1, 1, 1, 0, 0]
```

This project is **not** multi-label classification.

This project is multiclass classification.

So the correct tools are:

```text
ImageFolder
CrossEntropyLoss
Softmax
Argmax
```

---

## 5. Dataset Overview

The dataset used in this project is a fruit classification dataset from Kaggle.

The dataset was downloaded using KaggleHub:

```python
kagglehub.dataset_download("utkarshsaxenadn/fruits-classification")
```

After downloading, the dataset was detected at:

```text
C:\Users\U-ser\.cache\kagglehub\datasets\utkarshsaxenadn\fruits-classification\versions\1
```

The script automatically detected these folders:

```text
Train:
Fruits Classification/train

Validation:
Fruits Classification/valid

Test:
Fruits Classification/test
```

The detected class names were:

```text
['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']
```

The dataset sizes were:

| Dataset | Number of Images |
| ------- | ---------------: |
| Training set | 9700 |
| Validation set | 200 |
| Test set | 100 |

The training set is used to train the model.

The validation set is used during training to check whether the model is improving.

The test set is used after training to evaluate the final saved model.

---

## 6. Project Workflow

The full workflow is:

```text
Kaggle Fruit Dataset
        ↓
Download using KaggleHub
        ↓
Find train / valid / test folders
        ↓
Load images using ImageFolder
        ↓
Apply image transformations
        ↓
Use pretrained ResNet18
        ↓
Replace final classifier layer
        ↓
Train the model
        ↓
Validate after each epoch
        ↓
Use early stopping
        ↓
Save the best model
        ↓
Evaluate on the test set
        ↓
Predict unknown fruit images
```

The project is separated into four main files:

```text
model_fruit_multiclass.py
training_fruit_multiclass.py
evaluate_fruit_multiclass.py
predict_fruit_multiclass.py
```

The purpose of each file is:

| File | Purpose |
| ---- | ------- |
| `model_fruit_multiclass.py` | Defines the ResNet18 transfer learning model |
| `training_fruit_multiclass.py` | Downloads the dataset, trains the model, validates it, and saves the best model |
| `evaluate_fruit_multiclass.py` | Loads the saved model and evaluates it on the test set |
| `predict_fruit_multiclass.py` | Loads the saved model and predicts one unknown image |

---

## 7. Why Use Transfer Learning?

At first, I trained a custom CNN from scratch.

That means the model started with random weights.

The model had to learn everything by itself:

```text
edges
colors
textures
fruit shapes
fruit patterns
```

The custom CNN reached around:

```text
70% validation accuracy
```

This was not bad, but the model was still not very strong.

Then I used transfer learning with pretrained ResNet18.

Transfer learning means:

```text
use a model that already learned useful image features
then adapt it to a new task
```

ResNet18 already learned general image features from a large dataset.

For example, it already learned patterns such as:

```text
edges
curves
colors
textures
object shapes
```

So the model does not need to learn everything from zero.

It only needs to learn how to use those features for the new fruit classes:

```text
Apple
Banana
Grape
Mango
Strawberry
```

After using transfer learning, the validation accuracy improved to around:

```text
87.00%
```

This shows that transfer learning helped a lot.

---

## 8. What is ResNet18?

ResNet18 is a convolutional neural network architecture.

The name means:

```text
Residual Network with 18 layers
```

ResNet uses residual connections.

A residual connection allows information to skip some layers.

This helps deeper models train more effectively.

In this project, I do not use ResNet18 to classify the original ImageNet classes.

Instead, I use ResNet18 as a feature extractor and replace the final classifier layer with a new layer for five fruit classes.

---

## 9. File 1: `model_fruit_multiclass.py`

This file defines the model.

Instead of using a custom CNN from scratch, it uses pretrained ResNet18.

Important imports:

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
```

`resnet18` loads the ResNet18 architecture.

`ResNet18_Weights` loads pretrained weights.

---

## 10. Loading Pretrained ResNet18

The model loads pretrained ResNet18 using:

```python
weights = ResNet18_Weights.DEFAULT

self.model = resnet18(weights=weights)
```

This means the model is not starting from random weights.

It already has useful pretrained knowledge.

This is different from the custom CNN, where all weights started randomly.

---

## 11. Freezing the Backbone

The model has an option:

```python
freeze_backbone=True
```

When the backbone is frozen, the pretrained feature extractor does not update during training.

The code is:

```python
if freeze_backbone:
    for parameter in self.model.parameters():
        parameter.requires_grad = False
```

This means:

```text
do not change the pretrained ResNet18 feature extractor
only train the new final classifier layer
```

This is useful for the first transfer learning step because it is faster and safer.

The model already knows general image features.

So I only train the last layer to classify fruits.

---

## 12. Replacing the Final Layer

The original ResNet18 was trained for ImageNet classes.

This project has only five fruit classes.

So the final layer must be replaced.

First, I get the input size of the original final layer:

```python
in_features = self.model.fc.in_features
```

Then I replace it with a new classifier:

```python
self.model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, num_classes)
)
```

For this dataset:

```text
num_classes = 5
```

So the final output shape is:

```text
[batch_size, 5]
```

Each output is a raw score for one class:

```text
Apple score
Banana score
Grape score
Mango score
Strawberry score
```

These raw scores are called logits.

---

## 13. Why No Softmax Inside the Model?

The model does not use softmax inside the `forward()` function.

The forward function simply returns the model output:

```python
def forward(self, x):
    return self.model(x)
```

This is correct because during training, the loss function is:

```python
nn.CrossEntropyLoss()
```

`CrossEntropyLoss` expects raw logits.

It internally handles the softmax-like calculation.

So the correct workflow is:

```text
Training:
model outputs logits
CrossEntropyLoss calculates the loss

Prediction:
model outputs logits
softmax converts logits into probabilities
```

---

## 14. File 2: `training_fruit_multiclass.py`

This file trains the fruit classifier.

It performs these steps:

```text
download dataset
find train / validation / test folders
load images
apply transforms
create DataLoader
create model
define loss function
define optimizer
train the model
validate the model
apply early stopping
save the best model
plot training curve
```

---

## 15. Dataset Download

The dataset is downloaded using:

```python
DATASET_SLUG = "utkarshsaxenadn/fruits-classification"
```

Then:

```python
dataset_path = kagglehub.dataset_download(DATASET_SLUG)
```

The script prints the local path after downloading.

Example:

```text
Downloaded dataset path:
C:\Users\U-ser\.cache\kagglehub\datasets\utkarshsaxenadn\fruits-classification\versions\1
```

---

## 16. Finding Train, Validation, and Test Folders

The script automatically finds the train, validation, and test folders.

The detected result was:

```text
Train: ...\Fruits Classification\train
Val  : ...\Fruits Classification\valid
Test : ...\Fruits Classification\test
```

This is helpful because Kaggle datasets may have different folder structures.

The script searches for common folder names such as:

```text
train
training
valid
validation
test
testing
```

---

## 17. ImageFolder

The training file uses:

```python
torchvision.datasets.ImageFolder()
```

ImageFolder expects a folder structure like this:

```text
train/
    Apple/
        image1.jpg
        image2.jpg

    Banana/
        image1.jpg
        image2.jpg

    Grape/
        image1.jpg
        image2.jpg
```

The folder name becomes the class label.

For example:

```text
train/Banana/image1.jpg
```

becomes:

```text
label = Banana
```

The class mapping was:

```text
{'Apple': 0, 'Banana': 1, 'Grape': 2, 'Mango': 3, 'Strawberry': 4}
```

So the model learns:

```text
0 = Apple
1 = Banana
2 = Grape
3 = Mango
4 = Strawberry
```

---

## 18. Image Transformations

The training transform is:

```python
transforms.Resize((IMG_SIZE, IMG_SIZE))
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomRotation(15)
transforms.ColorJitter(...)
transforms.ToTensor()
transforms.Normalize(...)
```

The image size is:

```text
224 × 224
```

This image size is used because pretrained ResNet18 works well with ImageNet-style image preprocessing.

---

## 19. Resize

```python
transforms.Resize((IMG_SIZE, IMG_SIZE))
```

This makes every image the same size:

```text
224 × 224
```

This is needed because all images in a batch must have the same shape.

---

## 20. Data Augmentation

The training transform includes:

```python
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomRotation(15)
transforms.ColorJitter(...)
```

These transformations randomly modify training images.

For example, training images may be:

```text
flipped
rotated
made brighter or darker
changed in color saturation
```

This helps the model generalize better.

The model should still recognize a banana even if the image is slightly rotated or the lighting changes.

---

## 21. ToTensor

```python
transforms.ToTensor()
```

This converts the image into a PyTorch tensor.

Before this step, the image is a PIL image.

After this step, the image becomes a tensor with shape:

```text
[3, 224, 224]
```

The `3` means RGB channels:

```text
red
green
blue
```

---

## 22. Normalize

The normalization used is:

```python
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

These are common normalization values for ImageNet pretrained models.

This is important because ResNet18 was pretrained using this type of preprocessing.

If the input image values are normalized differently, the pretrained model may not work as well.

---

## 23. DataLoader

The DataLoader is created like this:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)
```

The batch size was:

```text
128
```

The training dataset had:

```text
9700 images
```

So one epoch had about:

```text
9700 / 128 ≈ 76 batches
```

This matches the training output:

```text
Epoch 1/25: 76/76
```

---

## 24. Device: CPU or GPU

The code checks whether GPU is available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

The output showed:

```text
Using device: cuda
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
```

So the model trained on GPU.

The model is moved to GPU using:

```python
model.to(device)
```

Each batch of images and labels is also moved to GPU:

```python
images = images.to(device)
labels = labels.to(device)
```

The model and data must be on the same device.

---

## 25. Loss Function

For multiclass classification, the loss function is:

```python
criterion = nn.CrossEntropyLoss()
```

This is the correct loss function because each image belongs to exactly one class.

The model outputs one score for each class.

Example:

```text
[Apple score, Banana score, Grape score, Mango score, Strawberry score]
```

The true label is one class index.

Example:

```text
Banana = 1
```

The loss tells the model how wrong the prediction was.

---

## 26. Optimizer

The optimizer used is Adam:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)
```

The optimizer updates the model weights.

The learning rate controls how large the weight updates are.

For transfer learning, a smaller learning rate is usually better.

In the final training run, the learning rate was:

```text
0.0003
```

This made training smoother than using a larger learning rate.

---

## 27. Learning Rate Scheduler

The training file uses:

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=3,
    factor=0.5
)
```

This scheduler watches the validation loss.

If the validation loss does not improve for several epochs, it reduces the learning rate.

This helps the model continue learning with smaller updates.

---

## 28. Training Loop

The main training loop repeats for several epochs.

Inside each epoch, the model trains on all training batches.

The main steps are:

```python
optimizer.zero_grad(set_to_none=True)

outputs = model(images)

loss = criterion(outputs, labels)

loss.backward()

optimizer.step()
```

These steps mean:

| Step | Code | Meaning |
| ---- | ---- | ------- |
| 1 | `optimizer.zero_grad()` | Clear old gradients |
| 2 | `outputs = model(images)` | Make predictions |
| 3 | `loss = criterion(outputs, labels)` | Compare predictions with true labels |
| 4 | `loss.backward()` | Calculate gradients |
| 5 | `optimizer.step()` | Update weights |

This is the standard PyTorch training process.

---

## 29. Mixed Precision Training

The training file uses:

```python
from torch.amp import autocast, GradScaler
```

and:

```python
with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
    outputs = model(images)
    loss = criterion(outputs, labels)
```

This is called mixed precision training.

It allows some calculations to use lower precision on GPU.

This can make training faster and use less GPU memory.

`GradScaler` helps keep training stable when mixed precision is used.

---

## 30. Validation Loop

After each training epoch, the model is evaluated on validation data.

The validation function uses:

```python
model.eval()
```

and:

```python
with torch.no_grad():
```

`model.eval()` sets the model to evaluation mode.

This turns off training-specific behavior such as dropout.

`torch.no_grad()` disables gradient calculation.

This makes validation faster and saves memory.

Validation data is not used to update the weights.

It is only used to check whether the model performs well on unseen images.

---

## 31. Early Stopping

Early stopping is an important part of the training process.

At the start, the code sets:

```python
best_val_loss = float("inf")
early_stop_counter = 0
```

After each epoch, the model checks whether the validation loss improved.

If the validation loss improves:

```python
if val_loss < best_val_loss:
```

then the model saves the new best weights:

```python
torch.save(model.state_dict(), MODEL_PATH)
```

and resets the counter:

```python
early_stop_counter = 0
```

If the validation loss does not improve, the counter increases:

```python
early_stop_counter += 1
```

If the counter reaches the patience value:

```python
if early_stop_counter >= PATIENCE:
    print("Early stopping triggered.")
    break
```

then training stops.

This prevents the model from training too long when validation performance is no longer improving.

---

## 32. Why Early Stopping Is Useful

Early stopping helps prevent unnecessary training.

It also helps protect the best model.

The final epoch is not always the best epoch.

So the script saves the model whenever validation loss improves.

The saved model file is:

```text
best_fruit_multiclass_model.pth
```

In this project, the best validation loss was:

```text
0.4066
```

The best validation accuracy reached:

```text
87.00%
```

So the saved model represents the best validation performance from the training process.

---

## 33. Final Training Result

The final training run used transfer learning with pretrained ResNet18 and a learning rate of:

```text
0.0003
```

The training result improved gradually.

Some important epochs were:

```text
Epoch 01 | Train Loss: 1.4428 | Train Acc: 39.06% | Val Loss: 1.0602 | Val Acc: 70.00%
Epoch 05 | Train Loss: 0.7199 | Train Acc: 75.33% | Val Loss: 0.5695 | Val Acc: 82.00%
Epoch 10 | Train Loss: 0.6168 | Train Acc: 78.35% | Val Loss: 0.4737 | Val Acc: 85.00%
Epoch 20 | Train Loss: 0.5672 | Train Acc: 79.28% | Val Loss: 0.4135 | Val Acc: 86.50%
Epoch 25 | Train Loss: 0.5557 | Train Acc: 79.75% | Val Loss: 0.4066 | Val Acc: 87.00%
```

The best validation result was:

| Metric | Value |
| ------ | ----: |
| Best Validation Loss | 0.4066 |
| Best Validation Accuracy | 87.00% |

This shows that transfer learning worked well.

---

## 34. Loss Curve Explanation

The loss curve shows that both training loss and validation loss decreased.

Training loss decreased from about:

```text
1.44 → 0.56
```

Validation loss decreased from about:

```text
1.06 → 0.41
```

This means the model learned useful fruit features.

The validation loss did not increase while the training loss decreased, so there is no strong sign of overfitting.

The validation loss was lower than the training loss.

This can happen because the training images are harder.

During training, data augmentation and dropout are active.

During validation, there is no random augmentation and dropout is turned off.

So validation images can be easier than training images.

Example image:

![Fruit Multiclass Loss Curve](/images/pytorch/fruit_multiclass_loss_curve.png)

---

## 35. File 3: `evaluate_fruit_multiclass.py`

This file evaluates the saved model.

It uses these saved files:

```text
best_fruit_multiclass_model.pth
fruit_class_names.json
fruit_multiclass_data_path.json
```

The evaluation file does these steps:

```text
load class names
load dataset path
load test dataset
load saved model
make predictions
calculate accuracy
print classification report
compare with dummy classifier
plot confusion matrix
```

---

## 36. Loading Class Names

The class names are saved during training in:

```text
fruit_class_names.json
```

The evaluation file loads them using:

```python
with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)
```

This is important because the model output index must match the correct class name.

Example:

```text
0 = Apple
1 = Banana
2 = Grape
3 = Mango
4 = Strawberry
```

---

## 37. Loading the Test Dataset

The evaluation file reads the dataset path from:

```text
fruit_multiclass_data_path.json
```

Because the dataset has a test folder, the evaluation file uses:

```text
Fruits Classification/test
```

The test set contains:

```text
100 images
```

The test data is not used for training.

It is used for final evaluation.

---

## 38. Loading the Saved Model

The evaluation file creates the same model architecture:

```python
model = FruitMulticlassCNN(num_classes=len(class_names)).to(device)
```

Then it loads the saved weights:

```python
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)
```

After loading, the model is set to evaluation mode:

```python
model.eval()
```

---

## 39. Making Predictions

During evaluation:

```python
outputs = model(images)
predictions = torch.argmax(outputs, dim=1)
```

The model outputs one score for each class.

The class with the highest score becomes the prediction.

Example:

```text
[1.2, 5.4, 0.3, -0.8, 0.2]
```

The highest score is at index 1.

So the model predicts:

```text
Banana
```

---

## 40. Final Evaluation Result

The final test result was:

```text
Accuracy: 82.00%
```

The test set contains 100 images.

So this means:

```text
82 images were predicted correctly
18 images were predicted incorrectly
```

The dummy classifier baseline was:

```text
20.00%
```

The dummy classifier always predicts the most common class.

Because the dataset has five balanced classes, the dummy classifier only gets around 20%.

This shows that the trained ResNet18 model learned useful image features.

The final result summary is:

| Metric | Value |
| ------ | ----: |
| Best Validation Accuracy | 87.00% |
| Best Validation Loss | 0.4066 |
| Test Accuracy | 82.00% |
| Dummy Classifier Accuracy | 20.00% |

---

## 41. Classification Report

The classification report was:

```text
              precision    recall  f1-score   support

       Apple       0.67      0.70      0.68        20
      Banana       0.79      0.95      0.86        20
       Grape       0.89      0.80      0.84        20
       Mango       0.88      0.70      0.78        20
  Strawberry       0.90      0.95      0.93        20

    accuracy                           0.82       100
   macro avg       0.83      0.82      0.82       100
weighted avg       0.83      0.82      0.82       100
```

The best classes were:

```text
Banana
Strawberry
Grape
```

The weakest class was:

```text
Apple
```

Apple had the lowest F1-score:

```text
0.68
```

This means Apple was more difficult for the model than the other classes.

---

## 42. Confusion Matrix Explanation

The confusion matrix was:

```text
[[14  2  1  1  2]
 [ 1 19  0  0  0]
 [ 2  1 16  1  0]
 [ 4  2  0 14  0]
 [ 0  0  1  0 19]]
```

The class order is:

```text
Apple
Banana
Grape
Mango
Strawberry
```

The diagonal values are correct predictions:

```text
Apple correctly predicted: 14
Banana correctly predicted: 19
Grape correctly predicted: 16
Mango correctly predicted: 14
Strawberry correctly predicted: 19
```

The total correct predictions are:

```text
14 + 19 + 16 + 14 + 19 = 82
```

Because there are 100 test images, the accuracy is:

```text
82 / 100 = 82%
```

Example image:

![Fruit Multiclass Confusion Matrix](/images/pytorch/fruit_multiclass_confusion_matrix.png)

---

## 43. Apple Performance

Apple had:

```text
14 correct predictions
6 wrong predictions
```

Apple was confused with:

```text
Banana: 2 images
Grape: 1 image
Mango: 1 image
Strawberry: 2 images
```

This means Apple is the weakest class in this result.

The model may confuse apples with other fruits because of similar colors, shapes, or backgrounds.

---

## 44. Banana Performance

Banana had:

```text
19 correct predictions
1 wrong prediction
```

Only one banana image was predicted as Apple.

Banana had strong recall:

```text
0.95
```

This means the model found most banana images correctly.

---

## 45. Grape Performance

Grape had:

```text
16 correct predictions
4 wrong predictions
```

Grape was confused with:

```text
Apple: 2 images
Banana: 1 image
Mango: 1 image
```

The model performed well on Grape, but it still made some mistakes.

---

## 46. Mango Performance

Mango had:

```text
14 correct predictions
6 wrong predictions
```

Mango was confused with:

```text
Apple: 4 images
Banana: 2 images
```

This shows that Mango and Apple are the most confusing pair.

This may happen because some mango and apple images have similar colors, round shapes, or backgrounds.

---

## 47. Strawberry Performance

Strawberry had:

```text
19 correct predictions
1 wrong prediction
```

Only one strawberry image was predicted as Grape.

Strawberry was the strongest class.

Its F1-score was:

```text
0.93
```

This means the model classified strawberries very well.

---

## 48. File 4: `predict_fruit_multiclass.py`

This file predicts one unknown fruit image.

The command is:

```bash
python3 predict_fruit_multiclass.py path/to/fruit_image.jpg
```

Example:

```bash
python3 predict_fruit_multiclass.py banana.jpg
```

The file does these steps:

```text
load class names
load saved model
open image
resize image
normalize image
convert image to tensor
make prediction
apply softmax
select highest probability
print prediction and confidence
```

---

## 49. Single Image Preprocessing

The image is opened using:

```python
image = Image.open(image_path).convert("RGB")
```

Then the same transform is applied:

```python
transforms.Resize((IMG_SIZE, IMG_SIZE))
transforms.ToTensor()
transforms.Normalize(...)
```

The image tensor originally has shape:

```text
[3, 224, 224]
```

But the model expects batch format:

```text
[batch_size, channels, height, width]
```

So the code adds one batch dimension:

```python
image_tensor = image_tensor.unsqueeze(0)
```

The final shape becomes:

```text
[1, 3, 224, 224]
```

This means:

```text
1 image
3 RGB channels
224 height
224 width
```

---

## 50. Softmax During Prediction

During prediction, the model outputs raw logits.

Then the code applies softmax:

```python
probabilities = torch.softmax(output, dim=1)
```

Softmax converts raw scores into probabilities.

Example:

```text
Apple: 2%
Banana: 91%
Grape: 1%
Mango: 4%
Strawberry: 2%
```

Then the highest probability is selected:

```python
confidence, predicted_class = torch.max(probabilities, dim=1)
```

If Banana has the highest probability, the final prediction is:

```text
Banana
```

---

## 51. Laptop Camera Testing with ROI

I also tested the model with the laptop camera.

The camera version uses a center ROI box.

The ROI version does this:

```text
draw a box in the center
use only the image inside the box
predict the fruit class
show confidence
hide prediction if confidence is below 60%
```

This is useful because the model should focus only on the fruit, not the whole room.

Without ROI, the model may use the full camera frame, including:

```text
face
wall
table
hand
keyboard
background
```

These background objects can confuse the model.

The ROI box makes the input more focused.

---

## 52. Confidence Threshold

The camera ROI file uses:

```python
CONFIDENCE_THRESHOLD = 0.60
```

This means:

```text
if confidence >= 60%, show prediction
if confidence < 60%, show "No confident fruit"
```

This helps avoid showing weak predictions.

However, this is not perfect.

Because this is a multiclass classifier, the model must always choose one of the known classes internally.

If there is no fruit, the model may still predict one of these:

```text
Apple
Banana
Grape
Mango
Strawberry
```

The threshold only hides low-confidence predictions.

If the model is confidently wrong, it may still show a fruit.

A better long-term solution is to add a new class:

```text
no_fruit
```

Then the model can learn:

```text
empty background → no_fruit
hand only → no_fruit
table only → no_fruit
face only → no_fruit
```

---

## 53. Testing with an Overripe Banana

I also tested the idea with an overripe banana image.

This is useful because an overripe banana may look different from normal training images.

It may have:

```text
black spots
brown skin
dark texture
unusual color
damaged shape
```

If the model still predicts Banana, it means the model learned more than just the clean yellow color.

It may have learned the banana shape and overall structure.

If the model fails, it may mean the model depends too much on the appearance of clean bananas.

This kind of test is useful for checking generalization.

---

## 54. Important Lessons

This project helped me understand a complete PyTorch multiclass image classification workflow.

Important lessons:

```text
1. Multiclass classification means one image has one correct class.
2. ImageFolder can automatically create labels from folder names.
3. ResNet18 can be used for transfer learning.
4. Pretrained models already know useful image features.
5. The final classifier layer must be replaced for the new number of classes.
6. CrossEntropyLoss is used for multiclass classification.
7. Softmax is used during prediction to convert logits into probabilities.
8. Early stopping saves the best model and prevents unnecessary training.
9. Validation loss is important for deciding the best model.
10. Data augmentation helps the model generalize.
11. A confusion matrix shows which classes are confused.
12. A confidence threshold can help in camera testing, but it is not the same as a no_fruit class.
```

---

## 55. Limitations

This project is useful for learning, but it has limitations.

First, the test set is small.

```text
Test dataset size = 100 images
```

Because the test set is small, the accuracy can change a lot with only a few wrong predictions.

Second, the model only knows five classes:

```text
Apple
Banana
Grape
Mango
Strawberry
```

If the input image is something else, the model will still choose one of these classes.

Third, the model does not have a `no_fruit` class.

So it cannot truly know when no fruit is present.

Fourth, the model does not locate the fruit.

It only classifies the image.

It does not draw bounding boxes.

If object location is needed, then an object detection model such as YOLO is better.

Fifth, real webcam images may look different from the training dataset.

This can reduce prediction accuracy.

---

## 56. Future Improvements

Possible improvements include:

```text
1. Fine-tune the full ResNet18 model.
2. Add a no_fruit class for camera testing.
3. Add more real webcam fruit images.
4. Add more overripe and damaged fruit images.
5. Increase the test set size.
6. Try MobileNetV2 for faster camera prediction.
7. Try EfficientNet for better accuracy.
8. Compare custom CNN with pretrained ResNet18.
9. Deploy the model using a simple web application.
10. Use YOLO if fruit location is needed.
```

The next useful improvement is to fine-tune the full ResNet18 model.

At first, the backbone was frozen:

```text
freeze_backbone=True
```

This means only the final classifier layer learned.

Later, I can set:

```text
freeze_backbone=False
```

Then the whole ResNet18 model can adjust to the fruit dataset.

For full fine-tuning, the learning rate should be smaller:

```text
0.0001
```

This may improve weak classes such as Apple and Mango.

---

## 57. Conclusion

In this project, I built a fruit multiclass image classifier using PyTorch and pretrained ResNet18.

The model classified fruit images into five classes:

```text
Apple
Banana
Grape
Mango
Strawberry
```

The custom CNN from scratch reached around:

```text
70% validation accuracy
```

After using transfer learning with pretrained ResNet18, the model reached:

```text
87.00% validation accuracy
82.00% test accuracy
```

The dummy classifier accuracy was only:

```text
20.00%
```

So the trained ResNet18 model learned useful image features.

The project introduced important deep learning concepts such as:

```text
ImageFolder
image transformations
ResNet18
pretrained weights
freezing the backbone
replacing the final layer
CrossEntropyLoss
softmax prediction
early stopping
confusion matrix
camera testing with ROI
```

Overall, this project is an important step after binary image classification because it shows how to build a stronger multiclass image classifier using transfer learning.

