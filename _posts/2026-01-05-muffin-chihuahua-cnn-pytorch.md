---
title: "Muffin vs Chihuahua Image Classification using PyTorch: A Complete CNN Binary Classification Project"
date: 2026-01-05
permalink: /posts/2026-01-05-muffin-chihuahua-cnn-pytorch
tags:

```
    - PyTorch
    - Deep Learning
    - CNN
    - Computer Vision
    - Image Classification
    - Binary Classification
    - Python
    - Torchvision
```

---

# Muffin vs Chihuahua Image Classification using PyTorch: A Complete CNN Binary Classification Project

## 1. Introduction

In the previous PyTorch binary classification project, I used a neural network to classify tabular data.

For example, in the breast cancer classification project, the model received numerical input features such as:

```text
mean radius
mean texture
mean perimeter
mean area
mean smoothness
```

Then the model predicted one of two possible classes:

```text
malignant
benign
```

That project was a **binary classification problem** because the output had only two classes.

In this project, I moved from tabular data to image data.

Instead of giving the model numerical features directly, I gave the model images.

The goal is to classify an image as either:

```text
chihuahua
```

or:

```text
muffin
```

This project is still a **binary classification problem**, but it is now a **computer vision problem**.

The model must learn visual patterns from images.

This project is useful because it shows a complete PyTorch computer vision workflow:

```text
download image dataset
load images from folders
apply image transformations
convert images to tensors
create DataLoader
build a CNN model
train the model
validate the model
save the best model
evaluate the model on test data
plot confusion matrix
compare with dummy classifier
predict a new unknown image
```

The important difference from the breast cancer project is that this project uses a **Convolutional Neural Network**, or CNN.

CNNs are designed for image data.

---

## 2. Project Goal

The main goal of this project is:

```text
Given an image,
predict whether it is a chihuahua or a muffin.
```

The model receives one image as input.

Then it outputs one prediction:

```text
0 = chihuahua
1 = muffin
```

The target meaning is:

| Target Value | Meaning   |
| -----------: | --------- |
|            0 | Chihuahua |
|            1 | Muffin    |

In simple words:

```text
If the image looks like a dog, predict chihuahua.
If the image looks like food, predict muffin.
```

This dataset is interesting because muffins and chihuahuas can sometimes look visually similar.

For example, both can have:

```text
round shapes
brown colors
dark spots
similar textures
```

So the model must learn meaningful image features instead of simply memorizing simple colors.

---

## 3. Why This Is an Image Classification Problem

In the breast cancer project, each sample was already represented as numbers.

Example:

```text
[14.2, 18.5, 92.1, 650.0, ...]
```

But in this project, each sample is an image.

A computer does not understand an image the same way humans do.

Humans see:

```text
a chihuahua
```

or:

```text
a muffin
```

But a computer sees pixel values.

For example, an image is stored as numbers.

For a grayscale image, each pixel may have a value from:

```text
0 to 255
```

where:

```text
0 = black
255 = white
```

For an RGB image, each pixel has three values:

```text
red
green
blue
```

So an image is actually a large array of numbers.

PyTorch converts the image into a tensor so that the neural network can process it.

---

## 4. Why Use CNN?

A normal fully connected neural network is not ideal for images.

For example, if an RGB image has size:

```text
128 × 128 × 3
```

then the total number of input values is:

```text
128 × 128 × 3 = 49,152
```

If we directly connect this to a fully connected layer, the model will have many parameters.

This is inefficient.

Images also have spatial structure.

For example, nearby pixels are related to each other.

A dog ear is made from a group of nearby pixels.

A muffin top is also made from local pixel patterns.

CNNs are good for images because they look at small local regions using filters.

CNNs can learn features such as:

```text
edges
curves
corners
textures
fur patterns
muffin texture
dog face shape
```

Then deeper layers combine these simple features into more meaningful patterns.

---

## 5. Project Workflow

The full workflow is:

```text
Image Dataset
        ↓
ImageFolder
        ↓
Image Transformations
        ↓
DataLoader
        ↓
CNN Model
        ↓
Loss Function
        ↓
Optimizer
        ↓
Training Loop
        ↓
Validation Loop
        ↓
Save Best Model
        ↓
Evaluation
        ↓
Prediction on New Image
```

In this project, I separated the code into four main files:

```text
model.py
training.py
evaluate.py
predict.py
```

The purpose of each file is:

| File          | Purpose                                |
| ------------- | -------------------------------------- |
| `model.py`    | Defines the CNN model architecture     |
| `training.py` | Trains and validates the model         |
| `evaluate.py` | Evaluates the saved model on test data |
| `predict.py`  | Predicts one new image                 |

This structure makes the project easier to understand and easier to maintain.

---

## 6. Dataset Structure

The dataset uses folders to represent classes.

The folder structure is:

```text
dataset/
    train/
        chihuahua/
            image1.jpg
            image2.jpg
            ...
        muffin/
            image1.jpg
            image2.jpg
            ...

    test/
        chihuahua/
            image1.jpg
            image2.jpg
            ...
        muffin/
            image1.jpg
            image2.jpg
            ...
```

This structure is important because PyTorch `ImageFolder` automatically uses folder names as class labels.

For example:

```text
train/chihuahua/image1.jpg
```

becomes:

```text
label = 0
```

and:

```text
train/muffin/image1.jpg
```

becomes:

```text
label = 1
```

The class mapping was:

```text
Classes: ['chihuahua', 'muffin']
Class to index: {'chihuahua': 0, 'muffin': 1}
```

So the model learns:

```text
0 = chihuahua
1 = muffin
```

---

## 7. Dataset Size

The dataset was split into training, validation, and test sets.

The sizes were:

| Dataset        | Number of Images |
| -------------- | ---------------: |
| Training set   |             3786 |
| Validation set |              947 |
| Test set       |             1184 |

The training set is used to update the model weights.

The validation set is used to check whether the model is improving on unseen data during training.

The test set is used after training to measure the final performance of the saved model.

---

## 8. Image Preprocessing

Before images are given to the CNN, they must be transformed.

The main transformation steps are:

```text
resize image
convert to grayscale
convert to tensor
normalize pixel values
```

The transformation code is similar to:

```python
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```

---

## 9. Resize

```python
transforms.Resize((32, 32))
```

This resizes every image to:

```text
32 × 32 pixels
```

CNN models require all images in a batch to have the same size.

If one image is 500 × 500 and another image is 300 × 200, they cannot be placed in the same tensor batch.

So all images are resized to the same shape.

For this beginner project, I used small images because they train faster.

The final image shape becomes:

```text
1 × 32 × 32
```

where:

```text
1 = grayscale channel
32 = height
32 = width
```

---

## 10. Grayscale

```python
transforms.Grayscale()
```

This converts the image from RGB to grayscale.

An RGB image has 3 channels:

```text
red
green
blue
```

The shape is:

```text
3 × 32 × 32
```

After grayscale conversion, the image has only 1 channel:

```text
1 × 32 × 32
```

This makes the model simpler.

The model can focus more on shape and texture instead of color.

For muffin vs chihuahua classification, grayscale can still work because the model can learn patterns such as:

```text
fur texture
muffin texture
dark spots
round shape
dog face pattern
```

---

## 11. ToTensor

```python
transforms.ToTensor()
```

This converts the image into a PyTorch tensor.

Before this step, the image is a PIL image.

After this step, it becomes a tensor that PyTorch can process.

Pixel values also change from:

```text
0 to 255
```

to:

```text
0.0 to 1.0
```

This is important because neural networks train better with smaller numerical values.

---

## 12. Normalize

```python
transforms.Normalize(mean=[0.5], std=[0.5])
```

This changes pixel values from approximately:

```text
0 to 1
```

to approximately:

```text
-1 to 1
```

This helps the neural network train more smoothly.

Neural networks usually train better when input values are centered around zero.

---

## 13. Data Augmentation

For training, I also used data augmentation.

Examples include:

```python
transforms.RandomHorizontalFlip()
transforms.RandomRotation()
```

Data augmentation randomly changes the training images.

For example:

```text
flip image left/right
rotate image slightly
```

This helps the model generalize better.

The model should learn that a muffin is still a muffin even if the image is slightly rotated.

The model should also learn that a chihuahua is still a chihuahua even if the image is flipped.

Data augmentation helps reduce overfitting.

---

## 14. DataLoader

After creating the dataset, I used DataLoader.

Example:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

The DataLoader gives images to the model in mini-batches.

Instead of training on one image at a time, the model trains on a group of images.

In this project:

```text
batch size = 512
```

This means the model processes up to 512 images at once.

The training set has 3786 images.

So one epoch has about:

```text
3786 / 512 ≈ 8 batches
```

Using batches makes training faster and more stable.

---

## 15. GPU Training

The model used GPU training.

The code checks for GPU using:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

If CUDA is available, the model uses the GPU.

The output showed:

```text
Using device: cuda
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
```

The model is moved to GPU using:

```python
model.to(device)
```

The images and labels are also moved to GPU during training:

```python
images = images.to(device)
labels = labels.to(device)
```

The model and data must be on the same device.

If the model is on GPU but the data is on CPU, PyTorch will produce an error.

---

## 16. Model File: `model.py`

The model file defines the CNN architecture.

The model has two main parts:

```text
feature extractor
classifier
```

The feature extractor learns image patterns.

The classifier uses those learned features to make the final decision.

Simplified code:

```python
import torch
import torch.nn as nn

class ImageClassificationModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)

        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(64, 1)

        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x
```

---

## 17. CNN Feature Extractor

The feature extractor uses convolution blocks.

Each block contains:

```text
Conv2d
BatchNorm2d
ReLU
MaxPool2d
```

The input image shape is:

```text
[B, 1, 32, 32]
```

where:

```text
B = batch size
1 = grayscale channel
32 = height
32 = width
```

After the first block:

```text
[B, 32, 16, 16]
```

After the second block:

```text
[B, 64, 8, 8]
```

After the third block:

```text
[B, 128, 4, 4]
```

So the model gradually changes the image into many feature maps.

---

## 18. Conv2d

```python
nn.Conv2d(1, 32, kernel_size=3, padding=1)
```

This layer receives a grayscale image with 1 input channel.

It produces 32 output feature maps.

Each feature map is produced by a filter.

A filter learns to detect a pattern.

For example, filters may learn to detect:

```text
edges
spots
curves
texture
```

The model learns these filters automatically during training.

---

## 19. BatchNorm2d

```python
nn.BatchNorm2d(32)
```

Batch normalization helps stabilize training.

It keeps feature values in a good range.

This can make training faster and smoother.

It can also help the model generalize better.

---

## 20. ReLU

```python
nn.ReLU()
```

ReLU is an activation function.

It changes negative values to zero and keeps positive values.

```text
ReLU(-3) = 0
ReLU(5) = 5
```

ReLU helps the model learn non-linear patterns.

Without activation functions, the neural network would be too simple.

---

## 21. MaxPool2d

```python
nn.MaxPool2d(2)
```

Max pooling reduces the size of the feature maps.

Example:

```text
32 × 32
↓
16 × 16
```

Pooling helps the model:

```text
reduce computation
keep important features
be less sensitive to small shifts
```

---

## 22. Flatten Layer

After the convolution blocks, the tensor shape is:

```text
[B, 128, 4, 4]
```

Before sending it to a fully connected layer, it must be flattened.

```python
nn.Flatten()
```

The flattened size is:

```text
128 × 4 × 4 = 2048
```

So each image becomes a vector of 2048 learned features.

---

## 23. Classifier

The classifier receives the 2048 features and makes the final prediction.

The classifier structure is:

```text
2048
↓
256
↓
64
↓
1
```

The final output has one value because this is binary classification.

That one value is called a **logit**.

---

## 24. Why One Output Node?

This is binary classification.

There are only two classes:

```text
0 = chihuahua
1 = muffin
```

So the model only needs one output value.

If the output logit is high, the model leans toward class 1.

If the output logit is low, the model leans toward class 0.

During evaluation, the logit is converted into a probability using sigmoid.

---

## 25. Loss Function

The loss function was:

```python
loss_fn = nn.BCEWithLogitsLoss()
```

This loss function is used for binary classification.

It combines:

```text
sigmoid
binary cross entropy loss
```

So during training, I do not put sigmoid inside the model.

This is the correct workflow:

```text
model outputs raw logits
BCEWithLogitsLoss handles sigmoid internally
```

During evaluation and prediction, I manually use sigmoid:

```python
probability = torch.sigmoid(output)
```

---

## 26. Optimizer

The optimizer was Adam:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

The optimizer updates the model weights.

The training process is:

```text
make prediction
calculate loss
calculate gradients
update weights
```

Adam is commonly used because it usually trains smoothly and quickly.

---

## 27. Training Loop

The training loop repeats for several epochs.

One epoch means the model sees the whole training dataset once.

The main training steps are:

```python
optimizer.zero_grad()

outputs = model(images)

loss = loss_fn(outputs, labels)

loss.backward()

optimizer.step()
```

These steps mean:

| Step | Code                              | Meaning                              |
| ---- | --------------------------------- | ------------------------------------ |
| 1    | `optimizer.zero_grad()`           | Clear old gradients                  |
| 2    | `outputs = model(images)`         | Make predictions                     |
| 3    | `loss = loss_fn(outputs, labels)` | Compare predictions with true labels |
| 4    | `loss.backward()`                 | Calculate gradients                  |
| 5    | `optimizer.step()`                | Update weights                       |

This is the basic PyTorch training process.

---

## 28. Validation Loop

After each training epoch, the model is evaluated on validation data.

Validation data is not used to update weights.

It is used to check whether the model is improving on unseen images.

During validation, the code uses:

```python
model.eval()
```

and:

```python
with torch.no_grad():
```

`model.eval()` sets the model to evaluation mode.

`torch.no_grad()` disables gradient calculation.

This makes evaluation faster and uses less memory.

---

## 29. Saving the Best Model

The training script saved the best model as:

```text
best_model.pth
```

The model was saved when the validation loss improved.

This is important because the final epoch is not always the best epoch.

In this project, the best validation performance happened around epoch 23.

The saved model was the best checkpoint, not simply the final model.

---

## 30. Training Result

The model trained successfully.

The validation accuracy reached around:

```text
88.17%
```

The training accuracy reached around:

```text
94%
```

This means the model learned strong visual features.

However, the training accuracy was higher than the validation accuracy.

This shows a small amount of overfitting.

Overfitting means the model performs very well on training images but not quite as well on unseen images.

This is normal in many computer vision projects.

The best model checkpoint helps reduce this issue because it saves the model with the best validation loss.

---

## 31. Evaluation File: `evaluate.py`

After training, I evaluated the model on the test dataset.

The evaluation process is:

```text
load test images
load saved model
make predictions
calculate metrics
plot confusion matrix
compare with dummy classifier
```

The model was loaded using:

```python
model.load_state_dict(
    torch.load("best_model.pth", map_location=device)
)
```

Then the model was set to evaluation mode:

```python
model.eval()
```

During testing, the model outputs logits.

The logits are converted to probabilities:

```python
probabilities = torch.sigmoid(outputs)
```

Then the class is selected using threshold 0.5:

```python
predictions = (probabilities > 0.5).long()
```

The meaning is:

```text
probability >= 0.5 → muffin
probability < 0.5  → chihuahua
```

---

## 32. Final Test Results

The final test results were:

| Metric    |  Value |
| --------- | -----: |
| Accuracy  | 86.57% |
| Precision | 86.53% |
| Recall    | 83.82% |
| F1-score  | 85.15% |

The model achieved:

```text
86.57% test accuracy
```

This means that out of 100 test images, the model correctly classified about 87 images.

---

## 33. Classification Report

The classification report was:

```text
              precision    recall  f1-score   support

   chihuahua       0.87      0.89      0.88       640
      muffin       0.87      0.84      0.85       544

    accuracy                           0.87      1184
   macro avg       0.87      0.86      0.86      1184
weighted avg       0.87      0.87      0.87      1184
```

This shows that the model performed well on both classes.

For chihuahua:

```text
precision = 0.87
recall = 0.89
f1-score = 0.88
```

For muffin:

```text
precision = 0.87
recall = 0.84
f1-score = 0.85
```

The model was slightly better at detecting chihuahuas than muffins.

---

## 34. Confusion Matrix

The confusion matrix was:

```text
[[569  71]
 [ 88 456]]
```

Because the class mapping is:

```text
0 = chihuahua
1 = muffin
```

the confusion matrix means:

| Actual Class                                    | Prediction Result |
| ----------------------------------------------- | ----------------- |
| 569 chihuahuas correctly predicted as chihuahua |                   |
| 71 chihuahuas wrongly predicted as muffin       |                   |
| 88 muffins wrongly predicted as chihuahua       |                   |
| 456 muffins correctly predicted as muffin       |                   |

The total correct predictions are:

```text
569 + 456 = 1025
```

The total test images are:

```text
1184
```

So the accuracy is:

```text
1025 / 1184 = 86.57%
```

This matches the evaluation result.

---

## 35. Dummy Classifier Baseline

A dummy classifier was used as a baseline.

The dummy classifier always predicts the most common class.

The dummy classifier accuracy was:

```text
54.05%
```

The CNN accuracy was:

```text
86.57%
```

This means the CNN performed much better than the baseline.

The dummy classifier does not learn image patterns.

It only predicts the majority class.

The CNN learned useful visual features from the images.

---

## 36. Prediction File: `predict.py`

After evaluation, I used the saved model to predict one new image.

The prediction file receives an image path:

```bash
python3 predict.py test_muffin.jpg
```

The image goes through the same preprocessing steps:

```text
resize to 32 × 32
convert to grayscale
convert to tensor
normalize
add batch dimension
```

A single image tensor originally has shape:

```text
[1, 32, 32]
```

But the model expects batch format:

```text
[B, C, H, W]
```

So the code adds one extra dimension:

```python
image_tensor = image_tensor.unsqueeze(0)
```

The final shape becomes:

```text
[1, 1, 32, 32]
```

This means:

```text
batch size = 1
channel = 1
height = 32
width = 32
```

---

## 37. Sigmoid During Prediction

The model outputs one raw logit.

The prediction file converts it to probability using sigmoid:

```python
probability_class_1 = torch.sigmoid(output).item()
```

Because class 1 is muffin:

```text
P(muffin) = probability_class_1
```

If:

```text
P(muffin) >= 0.5
```

then the prediction is:

```text
muffin
```

If:

```text
P(muffin) < 0.5
```

then the prediction is:

```text
chihuahua
```

Example output:

```text
Prediction : muffin
Confidence : 97.42%
```

or:

```text
Prediction : chihuahua
Confidence : 95.18%
```

---

## 38. Testing With a New Muffin Image

After training and evaluation, I tested the model with a new muffin image.

The model successfully classified the image as muffin.

This shows that the model can make predictions on new images outside the training loop.

However, one image test is not enough to prove perfect real-world performance.

A model should be tested with many new images under different conditions.

---

## 39. Why the Model Is Not Perfect

The test accuracy was good, but not 100%.

This is normal.

The dataset is intentionally challenging because muffins and chihuahuas can look similar.

The model may make mistakes when:

```text
the muffin has dark spots like eyes
the chihuahua face is round
the image is blurry
lighting is unusual
the object is cropped
the background is confusing
```

Computer vision models learn from pixel patterns.

If two classes share similar patterns, mistakes can happen.

---

## 40. Overfitting

In this project, there was a small sign of overfitting.

The training accuracy became higher than the validation accuracy.

For example:

```text
Training accuracy: around 94%
Validation accuracy: around 86–88%
```

This means the model learned the training images very well, but it did not perform equally well on unseen validation images.

In simple words:

```text
Good learning:
The model learns what muffins and chihuahuas generally look like.

Overfitting:
The model memorizes the training photos too much.
```

To reduce overfitting, I used:

```text
data augmentation
dropout
batch normalization
validation set
best model checkpoint
```

These techniques helped the model generalize better.

---

## 41. Comparison With the Original Simple CNN

The original simple CNN achieved around:

```text
78.55% accuracy
```

After improving the model with more convolution layers, batch normalization, dropout, GPU training, and better DataLoader settings, the final model achieved:

```text
86.57% accuracy
```

So the improved CNN gave a clear performance improvement.

The improvement was approximately:

```text
86.57% - 78.55% = 8.02%
```

This shows that model architecture and training strategy matter.

---

## 42. Important Lessons From This Project

This project helped me understand the full PyTorch image classification workflow.

Important lessons:

```text
1. Images must be converted into tensors before training.
2. CNNs are better than normal neural networks for image data.
3. ImageFolder can automatically create labels from folder names.
4. Image transformations are very important.
5. Data augmentation helps reduce overfitting.
6. DataLoader helps train using mini-batches.
7. BCEWithLogitsLoss is suitable for binary image classification.
8. Sigmoid is used during evaluation and prediction.
9. Accuracy alone is not enough.
10. Confusion matrix shows where the model makes mistakes.
11. Dummy classifier gives a useful baseline.
12. Saving the best model is better than saving only the final epoch.
```

---

## 43. Limitations

This project is useful for learning, but it has some limitations.

First, the image size is small:

```text
32 × 32
```

Small images train faster, but some visual details are lost.

Second, the model was trained from scratch.

A pretrained model such as ResNet18 might perform better.

Third, the dataset may not represent all real-world cases.

For example, images from different cameras, lighting conditions, or backgrounds may be harder.

Fourth, the model only predicts two classes:

```text
chihuahua
muffin
```

If we give it an image of something else, the model will still force the image into one of these two classes.

This is a common limitation of closed-set classification.

---

## 44. Future Improvements

Possible improvements include:

```text
1. Use larger image size such as 64 × 64 or 128 × 128.
2. Use stronger data augmentation.
3. Use transfer learning with ResNet18.
4. Train for more epochs with early stopping.
5. Add more diverse training images.
6. Try RGB images instead of grayscale.
7. Compare CNN from scratch with pretrained models.
8. Deploy the model using a simple web app.
9. Test the model with many real-world images.
10. Export the model to ONNX for deployment.
```

The most useful next step is transfer learning.

A pretrained model has already learned general image features from a large dataset.

Then we can fine-tune it for:

```text
chihuahua vs muffin
```

This may improve accuracy.

---

## 45. Conclusion

In this project, I built a complete binary image classification model using PyTorch.

The model learned to classify images as:

```text
chihuahua
```

or:

```text
muffin
```

The project started from image loading and ended with prediction on a new image.

The final model achieved:

```text
86.57% test accuracy
```

The dummy classifier achieved only:

```text
54.05% accuracy
```

So the CNN learned useful visual patterns from the images.

This project is an important step after tabular binary classification because it introduces computer vision concepts such as:

```text
image tensors
CNNs
convolution filters
feature maps
pooling
image augmentation
confusion matrix for image classification
prediction on unknown images
```

Overall, this project helped me understand how PyTorch can be used not only for numerical datasets, but also for real image classification tasks.
