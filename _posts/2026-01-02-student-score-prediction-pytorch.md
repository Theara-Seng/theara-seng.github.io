---

title: "Student Exam Score Prediction using PyTorch Linear Regression"
date: 2026-01-02
permalink: /posts/2026-01-02-student-score-prediction-pytorch
tags:

    - PyTorch
    - Machine Learning
    - Linear Regression
    - Student Performance
    - Deep Learning
    - Python

---

# Student Exam Score Prediction using PyTorch Linear Regression

## 1. Project Overview

This project focuses on building a simple machine learning model using **PyTorch** to predict a student's exam score.

The goal is to use student-related information such as study hours, attendance, sleep hours, course, study method, internet access, and exam difficulty to predict the final exam score.

This is a **regression problem** because the output is a numerical value.

```text
Input: student information
Output: predicted exam score
```

For example:

```text
Age: 20
Study hours: 4.5
Class attendance: 85%
Sleep hours: 7
Study method: online videos
Exam difficulty: moderate

Predicted exam score: 67.95 / 100
```

The project follows a common machine learning workflow:

```text
1. Load dataset
2. Prepare the data
3. Convert categorical data into numerical data
4. Scale the input features
5. Train a PyTorch linear regression model
6. Evaluate the model
7. Save the trained model
8. Load the model and predict new student scores
```

---

## 2. Dataset Information

The dataset used in this project is the **Exam Score Prediction Dataset** from Kaggle.

The dataset contains:

```text
Number of rows: 20,000
Number of original columns: 13
Target column: exam_score
```

The first few rows of the dataset look like this:

| student_id | age | gender | course  | study_hours | class_attendance | internet_access | sleep_hours | sleep_quality | study_method  | facility_rating | exam_difficulty | exam_score |
| ---------: | --: | ------ | ------- | ----------: | ---------------: | --------------- | ----------: | ------------- | ------------- | --------------- | --------------- | ---------: |
|          1 |  17 | male   | diploma |        2.78 |             92.9 | yes             |         7.4 | poor          | coaching      | low             | hard            |       58.9 |
|          2 |  23 | other  | bca     |        3.37 |             64.8 | yes             |         4.6 | average       | online videos | medium          | moderate        |       54.8 |
|          3 |  22 | male   | b.sc    |        7.88 |             76.8 | yes             |         8.5 | poor          | coaching      | high            | moderate        |       90.3 |
|          4 |  20 | other  | diploma |        0.67 |             48.4 | yes             |         5.8 | average       | online videos | low             | moderate        |       29.7 |
|          5 |  20 | female | diploma |        0.89 |             71.6 | yes             |         9.8 | poor          | coaching      | low             | moderate        |       43.7 |

---

## 3. Dataset Columns

The dataset contains the following columns:

| Column             | Type         | Description                             |
| ------------------ | ------------ | --------------------------------------- |
| `student_id`       | Numerical ID | Unique ID for each student              |
| `age`              | Numerical    | Student age                             |
| `gender`           | Categorical  | Student gender                          |
| `course`           | Categorical  | Student course or program               |
| `study_hours`      | Numerical    | Number of study hours                   |
| `class_attendance` | Numerical    | Attendance percentage                   |
| `internet_access`  | Categorical  | Whether the student has internet access |
| `sleep_hours`      | Numerical    | Number of sleeping hours                |
| `sleep_quality`    | Categorical  | Sleep quality level                     |
| `study_method`     | Categorical  | Method used for studying                |
| `facility_rating`  | Categorical  | Rating of study facilities              |
| `exam_difficulty`  | Categorical  | Difficulty level of the exam            |
| `exam_score`       | Numerical    | Final exam score                        |

The target column is:

```text
exam_score
```

This is the value that the model tries to predict.

---

## 4. Why This Is a Regression Problem

Machine learning problems can be divided into different types. Two common types are:

```text
Classification: predicting a category
Regression: predicting a number
```

This project is a regression problem because the output is a number.

Example:

```text
Predicted exam score = 67.95
```

The model is not predicting categories such as:

```text
Pass / Fail
Low / Medium / High
```

Instead, it predicts a continuous score from approximately 0 to 100.

---

## 5. Data Preparation

Before training the model, the raw dataset needs to be prepared.

Raw data cannot be used directly because it contains:

```text
1. Categorical text values
2. An ID column that is not useful
3. Features with different numerical ranges
4. Target and input columns mixed together
```

The data preparation file is named:

```text
DataPrep.py
```

---

## 6. Data Preparation Code

```python
#%% packages
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import kagglehub

#%% Download dataset
path = kagglehub.dataset_download("kundanbedmutha/exam-score-prediction-dataset")
print("Path to dataset files:", path)

#%% Find CSV file automatically
csv_files = [file for file in os.listdir(path) if file.endswith(".csv")]
print("CSV files found:", csv_files)

if len(csv_files) == 0:
    raise FileNotFoundError("No CSV file found in the dataset folder.")

full_path = os.path.join(path, csv_files[0])
print("Using CSV file:", full_path)

#%% Import data
student = pd.read_csv(full_path)

#%% Check data
print("\nFirst 5 rows:")
print(student.head())

print("\nColumns:")
print(student.columns)

print("\nDataset shape:")
print(student.shape)

print("\nMissing values:")
print(student.isnull().sum())

#%% Drop missing values
student = student.dropna()

#%% Drop ID column because it is not useful for prediction
if "student_id" in student.columns:
    student = student.drop(columns=["student_id"])

#%% Target column
target_column = "exam_score"

print("\nTarget column selected:", target_column)

#%% One-hot encoding
student_dummies = pd.get_dummies(student, drop_first=True, dtype=int)

print("\nShape after one-hot encoding:")
print(student_dummies.shape)

#%% Separate X and y
X_df = student_dummies.drop(columns=[target_column])
y_df = student_dummies[[target_column]]

# Save feature column names for prediction later
feature_columns = X_df.columns

# Convert to numpy
X = np.array(X_df, dtype=np.float32)
y = np.array(y_df, dtype=np.float32)

print("\nX shape:", X.shape)
print("y shape:", y.shape)

#%% Scale X
scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

print("\nData preparation completed.")
print("Final X shape:", X.shape)
print("Final y shape:", y.shape)
print("Number of input features:", X.shape[1])
```

---

## 7. Explanation of Data Preparation

### 7.1 Loading the Dataset

The dataset is downloaded using:

```python
path = kagglehub.dataset_download("kundanbedmutha/exam-score-prediction-dataset")
```

This downloads the dataset into the local Kaggle cache folder.

The code then automatically finds the CSV file:

```python
csv_files = [file for file in os.listdir(path) if file.endswith(".csv")]
```

This is useful because different Kaggle datasets may have different CSV filenames.

For this project, the CSV file found was:

```text
Exam_Score_Prediction.csv
```

---

### 7.2 Dropping `student_id`

The column `student_id` is removed:

```python
student = student.drop(columns=["student_id"])
```

This is important because `student_id` is only an identification number.

It does not describe the student's learning behavior.

For example:

```text
student_id = 1
student_id = 2
student_id = 3
```

These numbers do not mean that student 3 is better than student 1.

If we keep this column, the model may learn meaningless patterns from the ID numbers. Therefore, it is better to remove it.

---

### 7.3 One-Hot Encoding

Some columns contain text values.

Examples:

```text
gender = male / female / other
course = diploma / bca / b.sc
internet_access = yes / no
sleep_quality = poor / average / good
study_method = coaching / online videos / self study
facility_rating = low / medium / high
exam_difficulty = easy / moderate / hard
```

PyTorch cannot directly process text values. It needs numerical values.

Therefore, we use one-hot encoding:

```python
student_dummies = pd.get_dummies(student, drop_first=True, dtype=int)
```

One-hot encoding converts categorical values into numerical columns.

Example:

Before one-hot encoding:

| gender |
| ------ |
| male   |
| female |
| other  |

After one-hot encoding:

| gender_male | gender_other |
| ----------: | -----------: |
|           1 |            0 |
|           0 |            0 |
|           0 |            1 |

The first category is dropped because of:

```python
drop_first=True
```

This avoids unnecessary duplicate information.

---

### 7.4 Separating Input and Output

The input features are stored in `X`.

The target output is stored in `y`.

```python
X_df = student_dummies.drop(columns=[target_column])
y_df = student_dummies[[target_column]]
```

In this project:

```text
X = all student information except exam_score
y = exam_score
```

After preparation, the shape becomes:

```text
X shape: (20000, 23)
y shape: (20000, 1)
```

This means:

```text
20,000 students
23 input features
1 target output
```

---

### 7.5 Saving Feature Columns

The code saves the input column names:

```python
feature_columns = X_df.columns
```

This is important for real prediction later.

When predicting a new student, the input must have the same columns in the same order as the training data.

If the order is different, the model will misunderstand the input.

Example:

```text
Training order:
age, study_hours, class_attendance, sleep_hours, gender_male, ...

Prediction order:
sleep_hours, age, gender_male, study_hours, ...
```

This would be wrong.

Therefore, the feature column list must be saved and reused during prediction.

---

### 7.6 Scaling the Input Features

The input features are scaled using:

```python
scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)
```

Scaling is important because the numerical values have different ranges.

Example:

```text
age: around 17 to 25
study_hours: around 0 to 10
class_attendance: around 0 to 100
sleep_hours: around 3 to 10
```

Without scaling, features with larger numbers may dominate the training process.

StandardScaler changes the input features so that they have approximately:

```text
mean = 0
standard deviation = 1
```

This helps the model train more smoothly.

---

## 8. Model Training

The training file is named:

```text
train_model.py
```

The model used is a simple **linear regression model** created with PyTorch.

The model tries to learn this relationship:

```text
exam_score = w1x1 + w2x2 + w3x3 + ... + b
```

Where:

```text
w = learned weights
x = input features
b = bias
```

Because there are 23 input features, the model learns:

```text
23 weights + 1 bias
```

---

## 9. Training Code

```python
#%% packages
import torch
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math

from DataPrep import X, y, scaler, feature_columns

#%% Hyperparameters
EPOCHS = 1000
LEARNING_RATE = 0.1

#%% Convert to tensor
X_tensor = torch.from_numpy(X.astype(np.float32))
y_tensor = torch.from_numpy(y.astype(np.float32))

print("X_tensor shape:", X_tensor.shape)
print("y_tensor shape:", y_tensor.shape)

#%% Model class
class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

#%% Create model
model = LinearRegression(input_size=X.shape[1], output_size=1)

# Start bias near average exam score
with torch.no_grad():
    model.linear.bias.fill_(float(y.mean()))

#%% Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.MSELoss()

#%% Train model
for epoch in range(EPOCHS):
    y_predict = model(X_tensor)

    loss = loss_fn(y_predict, y_tensor)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

#%% Evaluate model
model.eval()

with torch.no_grad():
    y_pred = model(X_tensor).detach().numpy()

r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = math.sqrt(mse)
mae = mean_absolute_error(y, y_pred)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R-squared: {r2:.2f}")

#%% Save model and preprocessing tools
torch.save(model.state_dict(), "student_score_model.pth")
joblib.dump(scaler, "student_scaler.pkl")
joblib.dump(feature_columns, "student_feature_columns.pkl")

print("--------------------------------")
print("Student score model saved successfully.")
print("Saved files:")
print("1. student_score_model.pth")
print("2. student_scaler.pkl")
print("3. student_feature_columns.pkl")
print("--------------------------------")
```

---

## 10. Explanation of Training Code

### 10.1 Importing Prepared Data

```python
from DataPrep import X, y, scaler, feature_columns
```

This imports the prepared data from `DataPrep.py`.

The training code does not need to load and clean the CSV file again. It simply receives:

```text
X = prepared input features
y = target exam scores
scaler = fitted StandardScaler
feature_columns = input column names
```

---

### 10.2 Converting NumPy Arrays to PyTorch Tensors

```python
X_tensor = torch.from_numpy(X.astype(np.float32))
y_tensor = torch.from_numpy(y.astype(np.float32))
```

PyTorch models use tensors.

So the NumPy arrays must be converted into PyTorch tensors.

The output shape is:

```text
X_tensor shape: torch.Size([20000, 23])
y_tensor shape: torch.Size([20000, 1])
```

This confirms that the model receives 20,000 samples, with 23 input features for each student.

---

### 10.3 Model Structure

```python
class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)
```

This is a simple PyTorch linear regression model.

The layer:

```python
torch.nn.Linear(input_size, output_size)
```

means:

```text
Input: 23 features
Output: 1 predicted exam score
```

---

### 10.4 Bias Initialization

```python
with torch.no_grad():
    model.linear.bias.fill_(float(y.mean()))
```

This line starts the model bias near the average exam score.

This is useful because exam scores are around 0 to 100.

If the bias starts near 0, the model may initially predict scores close to 0, causing a very large loss.

By starting near the average exam score, the model trains faster.

Before bias initialization, the first loss was very large:

```text
Epoch 0 Loss ≈ 4257
```

After bias initialization, the first loss became much smaller:

```text
Epoch 0 Loss ≈ 354
```

This makes training more stable and efficient.

---

### 10.5 Loss Function

```python
loss_fn = torch.nn.MSELoss()
```

The model uses **Mean Squared Error**.

MSE calculates the average squared difference between the predicted score and the actual score.

Example:

```text
Actual score = 80
Predicted score = 70
Error = 10
Squared error = 100
```

Since the error is squared, MSE values can look large.

---

### 10.6 Optimizer

```python
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
```

The optimizer updates the model weights.

Adam is a commonly used optimizer because it usually trains faster and more smoothly than simple gradient descent.

---

### 10.7 Training Loop

```python
for epoch in range(EPOCHS):
    y_predict = model(X_tensor)
    loss = loss_fn(y_predict, y_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

Each epoch follows this process:

| Step | Code                          | Meaning                              |
| ---- | ----------------------------- | ------------------------------------ |
| 1    | `y_predict = model(X_tensor)` | Make predictions                     |
| 2    | `loss = loss_fn(...)`         | Compare predictions with true scores |
| 3    | `loss.backward()`             | Calculate gradients                  |
| 4    | `optimizer.step()`            | Update model weights                 |
| 5    | `optimizer.zero_grad()`       | Clear gradients for next epoch       |

---

## 11. Training Results

The model was trained with:

```text
Epochs: 1000
Learning rate: 0.1
Optimizer: Adam
Loss function: MSELoss
```

The training result was:

```text
Epoch 0, Loss: 354.3196
Epoch 50, Loss: 174.4032
Epoch 100, Loss: 121.0813
Epoch 150, Loss: 102.2507
Epoch 200, Loss: 96.9915
Epoch 250, Loss: 95.9081
Epoch 300, Loss: 95.7452
Epoch 350, Loss: 95.7274
Epoch 400, Loss: 95.7260
Epoch 450, Loss: 95.7260
Epoch 500, Loss: 95.7260
Epoch 550, Loss: 95.7260
Epoch 600, Loss: 95.7260
Epoch 650, Loss: 95.7260
Epoch 700, Loss: 95.7260
Epoch 750, Loss: 95.7260
Epoch 800, Loss: 95.7260
Epoch 850, Loss: 95.7260
Epoch 900, Loss: 95.7260
Epoch 950, Loss: 95.7260
```

---

## 12. Loss Graph Analysis

The training loss decreased from:

```text
354.3196 → 95.7260
```

This means the model successfully learned patterns from the data.

A possible loss graph would look like this:

```text
Loss
350 | *
300 |
250 |
200 |       *
150 |             *
100 |                    ***************
 50 |
  0 +-----------------------------------
       0   100   200   300   400   1000
                 Epoch
```

### Interpretation

The loss decreases quickly during the first 200 epochs.

After around epoch 300 to 400, the loss becomes almost flat.

This means the model has mostly finished learning. Additional epochs after this point do not improve the model much.

### Important Observation

From epoch 400 to epoch 950, the loss stays almost the same:

```text
Loss ≈ 95.7260
```

This shows that the model has converged.

Therefore, 1000 epochs are not strictly necessary. A smaller value such as 400 or 500 epochs may be enough.

---

## 13. Why the Loss Still Looks Large

The final loss is:

```text
MSE = 95.7260
```

This may look large, but it is not as bad as it seems.

The loss function is MSE, which means the errors are squared.

To make the result easier to understand, we calculate RMSE:

```text
RMSE = sqrt(MSE)
RMSE = sqrt(95.7260)
RMSE ≈ 9.78
```

This means the model's typical prediction error is about:

```text
±9.78 exam score points
```

For an exam score range of 0 to 100, this is reasonable for a simple linear regression model.

---

## 14. Model Evaluation Metrics

The final model achieved:

| Metric    | Value | Meaning                                   |
| --------- | ----: | ----------------------------------------- |
| MSE       | 95.73 | Average squared error                     |
| RMSE      |  9.78 | Typical prediction error in score points  |
| R-squared |  0.73 | The model explains 73% of score variation |

The most important result is:

```text
R-squared = 0.73
```

This means the model explains about 73% of the variation in exam scores.

For a simple linear regression model, this is a good result.

---

## 15. R-squared Explanation

R-squared shows how well the model explains the target value.

```text
R² = 0 means the model explains nothing.
R² = 1 means the model explains everything perfectly.
```

In this project:

```text
R² = 0.73
```

This means:

```text
The model explains around 73% of the variation in exam scores.
```

The remaining 27% may be caused by other factors not included in the dataset, such as:

```text
student motivation
teacher quality
exam preparation strategy
health condition
difficulty of specific topics
random variation
```

---

## 16. Saving the Model

After training, three files are saved:

```text
student_score_model.pth
student_scaler.pkl
student_feature_columns.pkl
```

### 16.1 `student_score_model.pth`

This file stores the trained PyTorch model weights.

It contains the learned values of:

```text
weights
bias
```

This file allows us to use the model later without training again.

### 16.2 `student_scaler.pkl`

This file stores the fitted StandardScaler.

It is needed because real student input must be scaled in the same way as the training data.

### 16.3 `student_feature_columns.pkl`

This file stores the exact input columns used during training.

It ensures that the new input data has the same format and column order as the training data.

---

## 17. Prediction Code

After training and saving the model, we can test new students without training again.

The prediction file is named:

```text
predict_real.py
```

```python
#%% packages
import torch
import numpy as np
import pandas as pd
import joblib

#%% Load saved preprocessing tools
scaler = joblib.load("student_scaler.pkl")
feature_columns = joblib.load("student_feature_columns.pkl")

#%% Model class
class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

#%% Create model structure
model = LinearRegression(
    input_size=len(feature_columns),
    output_size=1
)

#%% Load trained model weights
model.load_state_dict(torch.load("student_score_model.pth"))
model.eval()

#%% Prediction function
def predict_exam_score(real_student):
    # Create empty input with the same columns as training data
    real_input = pd.DataFrame(
        np.zeros((1, len(feature_columns))),
        columns=feature_columns
    )

    # Fill numerical columns directly
    for key, value in real_student.items():
        if key in real_input.columns:
            real_input.loc[0, key] = value

    # Fill categorical one-hot columns manually
    for key, value in real_student.items():
        dummy_column = f"{key}_{value}"

        if dummy_column in real_input.columns:
            real_input.loc[0, dummy_column] = 1

    # Scale using saved scaler
    real_scaled = scaler.transform(real_input.values).astype(np.float32)

    # Convert to tensor
    real_tensor = torch.from_numpy(real_scaled)

    # Predict
    with torch.no_grad():
        prediction = model(real_tensor)

    # Keep prediction between 0 and 100
    score = prediction.item()
    score = max(0, min(100, score))

    return score
```

---

## 18. Explanation of Prediction Code

### 18.1 Loading the Saved Model

```python
model.load_state_dict(torch.load("student_score_model.pth"))
model.eval()
```

This loads the trained model weights.

The model does not train again. It only uses the saved knowledge from the training process.

---

### 18.2 Creating Empty Input

```python
real_input = pd.DataFrame(
    np.zeros((1, len(feature_columns))),
    columns=feature_columns
)
```

This creates an empty input row with the same columns used during training.

This is necessary because the model expects exactly 23 input features.

---

### 18.3 Filling Numerical Columns

```python
for key, value in real_student.items():
    if key in real_input.columns:
        real_input.loc[0, key] = value
```

This fills numerical features such as:

```text
age
study_hours
class_attendance
sleep_hours
```

---

### 18.4 Filling Categorical Columns

```python
dummy_column = f"{key}_{value}"
```

This creates the one-hot column name for categorical values.

Example:

```text
gender = female
```

becomes:

```text
gender_female
```

If this column exists in the training feature columns, the code sets it to 1.

---

### 18.5 Scaling Real Input

```python
real_scaled = scaler.transform(real_input.values).astype(np.float32)
```

The real student input must be scaled using the same scaler used during training.

This is important because the model was trained on scaled data.

---

### 18.6 Prediction

```python
with torch.no_grad():
    prediction = model(real_tensor)
```

The model predicts the exam score.

`torch.no_grad()` is used because we are not training the model. We only want prediction.

---

### 18.7 Clamping the Score

```python
score = max(0, min(100, score))
```

This keeps the predicted score between 0 and 100.

This is needed because linear regression can predict values outside the realistic range.

For example, the model may predict:

```text
110.53
```

But exam scores should not exceed 100, so the value is capped at:

```text
100.00
```

---

## 19. Prediction Testing

Three test students were used.

---

### 19.1 Student Test 1: Average Student

```python
student_1 = {
    "age": 20,
    "gender": "female",
    "course": "diploma",
    "study_hours": 4.5,
    "class_attendance": 85.0,
    "internet_access": "yes",
    "sleep_hours": 7.0,
    "sleep_quality": "average",
    "study_method": "online videos",
    "facility_rating": "medium",
    "exam_difficulty": "moderate"
}
```

Prediction:

```text
Predicted Exam Score: 67.95 / 100
```

### Interpretation

This student has moderate study hours, good attendance, average sleep quality, and moderate exam difficulty.

The predicted score of 67.95 suggests average to good performance.

---

### 19.2 Student Test 2: Strong Student

```python
student_2 = {
    "age": 22,
    "gender": "male",
    "course": "b.sc",
    "study_hours": 8.0,
    "class_attendance": 95.0,
    "internet_access": "yes",
    "sleep_hours": 8.0,
    "sleep_quality": "good",
    "study_method": "coaching",
    "facility_rating": "high",
    "exam_difficulty": "moderate"
}
```

Raw prediction:

```text
Predicted Exam Score: 110.53 / 100
```

After clamping:

```text
Predicted Exam Score: 100.00 / 100
```

### Interpretation

This student has strong study habits, high attendance, good sleep, coaching support, and high facility rating.

The model predicts a very high score. Since linear regression can produce values above 100, the final score is capped at 100.

---

### 19.3 Student Test 3: Weak Student

```python
student_3 = {
    "age": 19,
    "gender": "other",
    "course": "bca",
    "study_hours": 1.0,
    "class_attendance": 45.0,
    "internet_access": "no",
    "sleep_hours": 4.5,
    "sleep_quality": "poor",
    "study_method": "self study",
    "facility_rating": "low",
    "exam_difficulty": "hard"
}
```

Prediction:

```text
Predicted Exam Score: 30.29 / 100
```

### Interpretation

This student has low study hours, low attendance, no internet access, poor sleep quality, low facility rating, and a hard exam.

The predicted score of 30.29 suggests weak performance.

---

## 20. Prediction Summary

| Student   | Description     | Predicted Score |
| --------- | --------------- | --------------: |
| Student 1 | Average student |     67.95 / 100 |
| Student 2 | Strong student  |    100.00 / 100 |
| Student 3 | Weak student    |     30.29 / 100 |

The predictions make sense:

```text
Better study habits and attendance → higher predicted score
Poor study habits and attendance → lower predicted score
```

---

## 21. Important Limitation: Linear Regression Can Exceed 100

One important limitation of linear regression is that it does not know the natural boundary of exam scores.

Exam scores should be between:

```text
0 and 100
```

But linear regression can output:

```text
negative values
values above 100
```

For example:

```text
Raw prediction = 110.53
```

This is not valid as an exam score, so the prediction is capped at 100.

This is a practical solution, but a better future improvement would be to use a model design or output transformation that naturally limits the output range.

---

## 22. Possible Graphs for the Blog Post

### 22.1 Loss Curve

The loss curve shows how the model improves during training.

The x-axis represents the epoch number.

The y-axis represents the loss value.

Expected pattern:

```text
Loss decreases quickly at first.
Loss becomes stable after around 400 epochs.
```

Suggested image name:

```markdown
![Training Loss Curve](/images/student-score/loss_curve.png)
```

### Graph Interpretation

If the loss curve goes down, the model is learning.

In this project, the loss decreased from:

```text
354.3196 to 95.7260
```

This shows successful training.

---

### 22.2 Actual vs Predicted Score Graph

This graph compares the actual exam scores with the predicted exam scores.

The x-axis represents predicted scores.

The y-axis represents actual scores.

Suggested image name:

```markdown
![Actual vs Predicted Exam Score](/images/student-score/actual_vs_predicted.png)
```

### Graph Interpretation

If the model is perfect, all points should lie close to a diagonal line.

If the points are widely scattered, the model has more prediction error.

Since the model achieved:

```text
R² = 0.73
```

we expect the graph to show a positive relationship between predicted and actual scores.

---

### 22.3 Feature Importance Graph

For a linear regression model, the learned weights can be used to understand which features influence the prediction.

A positive weight means the feature increases the predicted exam score.

A negative weight means the feature decreases the predicted exam score.

Suggested image name:

```markdown
![Feature Importance](/images/student-score/feature_importance.png)
```

### Graph Interpretation

Possible positive features may include:

```text
study_hours
class_attendance
sleep_hours
good facility rating
good study method
```

Possible negative features may include:

```text
hard exam difficulty
poor sleep quality
low facility rating
```

The exact feature importance depends on the trained model weights.

---

## 23. Code to Generate Graphs

The following code can be added to `train_model.py` to generate useful graphs.

```python
import matplotlib.pyplot as plt
import pandas as pd

#%% Plot loss curve
plt.figure(figsize=(8, 5))
plt.plot(loss_list)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.savefig("loss_curve.png", dpi=300, bbox_inches="tight")
plt.show()

#%% Plot actual vs predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_pred.flatten(), y.flatten(), alpha=0.3, s=10)
plt.title("Actual vs Predicted Exam Scores")
plt.xlabel("Predicted Exam Score")
plt.ylabel("Actual Exam Score")
plt.grid(True)
plt.savefig("actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.show()

#%% Feature importance
weights = model.linear.weight.detach().numpy().flatten()

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "weight": weights
})

importance_df["absolute_weight"] = importance_df["weight"].abs()
importance_df = importance_df.sort_values("absolute_weight", ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(importance_df["feature"], importance_df["weight"])
plt.title("Feature Importance from Linear Regression Weights")
plt.xlabel("Weight Value")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.grid(True)
plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()
```

---

## 24. Limitations of This Project

Although the model performs well, there are some limitations.

### 24.1 Evaluation Uses Training Data

The current R² score was calculated on the same data used for training.

This means the result is not a true test of unseen performance.

A better workflow is to split the dataset into:

```text
training set
validation set
test set
```

Then evaluate the model on data that it has never seen.

---

### 24.2 Linear Regression Is Simple

The model used in this project is linear regression.

It assumes the relationship between inputs and exam score is mostly linear.

However, student performance may be more complex.

For example:

```text
Studying more helps, but only up to a certain point.
Too little sleep is bad, but too much sleep may not always improve performance.
Exam difficulty may interact with study hours.
```

A simple linear model may not fully capture these patterns.

---

### 24.3 Prediction Can Exceed Valid Range

Because linear regression has no output limit, it can predict values below 0 or above 100.

This is why the prediction function clamps the output between 0 and 100.

---

### 24.4 Dataset May Not Represent All Students

The dataset may not represent all real-world students.

Real student performance may also depend on factors not included in the dataset, such as:

```text
motivation
mental health
teacher quality
family support
learning style
previous knowledge
exam stress
```

---

## 25. Possible Improvements

This project can be improved in several ways.

### 25.1 Train/Validation/Test Split

Instead of training and testing on the same data, split the data into:

```text
70% training
15% validation
15% testing
```

This gives a better estimate of real-world performance.

---

### 25.2 Use a Neural Network

A small neural network may capture nonlinear relationships better than linear regression.

Example:

```python
class StudentScoreNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(StudentScoreNN, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.network(x)
```

This model may improve prediction performance because it can learn more complex patterns.

---

### 25.3 Add More Features

The model may improve if the dataset includes more useful features such as:

```text
previous exam score
assignment completion
quiz score
hours spent on revision
number of missed classes
teacher feedback
motivation level
stress level
```

---

### 25.4 Better Output Control

Instead of manually clamping the prediction, another approach is to design the model output so that it naturally stays between 0 and 100.

For example, a sigmoid output could be scaled:

```text
score = sigmoid(output) × 100
```

---

## 26. Final Conclusion

This project successfully built a student exam score prediction model using PyTorch linear regression.

The model used student information such as age, course, study hours, attendance, sleep hours, study method, facility rating, and exam difficulty to predict exam scores.

The final model achieved:

```text
MSE: 95.73
RMSE: 9.78
R-squared: 0.73
```

This means the model explains about 73% of the variation in exam scores, and the typical prediction error is about 9.78 score points.

The prediction examples also show reasonable behavior:

```text
Average student → 67.95 / 100
Strong student → 100.00 / 100
Weak student → 30.29 / 100
```

Overall, this project is a useful beginner-friendly example of how to use PyTorch for regression, data preprocessing, model training, model saving, and real-world prediction.

It also demonstrates the complete machine learning workflow:

```text
Dataset → Preprocessing → Training → Evaluation → Saving → Prediction
```
