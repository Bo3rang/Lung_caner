# **✅ Tổng quan nội dung trong bài báo:**

## **1. Datasets**

- **JSRT Dataset** (247 ảnh):

  - 100 ảnh ung thư (malignant)
  - 54 ảnh lành tính (benign)
  - 93 ảnh không có nodule

- **ChestX-ray14 Dataset** (\~112,120 ảnh):

  - Có 14 bệnh, bao gồm "Nodule"
  - Dùng để **huấn luyện mô hình phân biệt nodule / non-nodule**

---

## **2. Các bước tiền xử lý ảnh**

- Tăng tương phản: Histogram Equalization
- Lọc nhiễu: Median Filter (3x3)
- Resize: 224x224
- Chuẩn hóa: theo mean/std của ImageNet

---

## **3. Mô hình và chiến lược Transfer Learning**

#### 🔷 Base Model

- DenseNet-121 (pretrained trên ImageNet)
- Layer cuối thay bằng 1 node sigmoid

#### 🔷 **Model A**

- Huấn luyện từ Base Model trên **ChestX-ray14**
- Phân biệt **nodule vs non-nodule**
- Dữ liệu lớn, học tốt, được coi là “chuyên gia phân biệt nodule”

#### 🔷 **Model B**

- Huấn luyện Base Model trên **JSRT**
- Phân biệt **lung cancer vs non-cancer** (malignant vs benign + no nodule)
- 10-fold cross-validation
- Tăng cường dữ liệu: xoay ±30°, lật ngang

#### 🔷 **Model C**

- Huấn luyện lại từ **Model A** trên **JSRT**
- Tức là **transfer 2 lần**: ImageNet → ChestX-ray14 → JSRT
- Mục tiêu: phân biệt malignant / non-malignant
- Kết quả tốt nhất trong bài báo

---

## **4 . Đánh giá**

| Model | Dataset      | Nhiệm vụ                           | Accuracy        | Specificity  | Sensitivity  |
| ----- | ------------ | ---------------------------------- | --------------- | ------------ | ------------ |
| A     | ChestX-ray14 | Nodule vs Non-nodule               | **84.02%**      | 85.34%       | 82.71%       |
| B     | JSRT         | Cancer vs Non-cancer               | 65.51±7.67%     | 80.95±20.59% | 45.67±21.36% |
| C     | ChestX+JSRT  | Cancer vs Non-cancer (fine-tune A) | **74.43±6.01%** | 74.96±9.85%  | 74.68±15.33% |

# **Các bước thực hiện chi tiết theo bài báo (viết lại theo môi trường Google Colab, sử dụng images\_001 duy nhất)**

## Phần 1: Chuẩn bị dữ liệu – ChestX-ray14 (Google Colab, chỉ dùng images\_001)

### Bước 1.1: Tải dữ liệu

- Upload file `Data_Entry_2017.csv` và thư mục `images_001` lên Google Drive.
- Mount Google Drive trong Colab:

```python
from google.colab import drive 
drive.mount('/content/drive')
```

### Bước 1.2: Gán nhãn và đọc thông tin ảnh

```python
import pandas as pd

csv_path = '/content/drive/MyDrive/ChestXray/Data_Entry_2017.csv'
df = pd.read_csv(csv_path)

# Gán nhãn binary: 1 nếu có Nodule, 0 nếu không
df['Label'] = df['Finding Labels'].apply(lambda x: 1 if 'Nodule' in x else 0)
print('Số ảnh Positive (label=1):', df[df['Label'] == 1].shape[0])
print('Số ảnh Negative (label=0):', df[df['Label'] == 0].shape[0])
```

### Bước 1.3: Chia train/val/test theo tỷ lệ bài báo (trên tập đã lọc sẵn)

```python
from sklearn.utils import shuffle

# Tách positive và negative
df_positive = df[df['Label'] == 1]
df_negative = df[df['Label'] == 0]

# Lấy mẫu nhỏ để demo trên images_001 (chỉnh số lượng tuỳ theo thực tế ảnh)
pos_train = df_positive.sample(n=200, random_state=42)
pos_val = df_positive.drop(pos_train.index).sample(n=50, random_state=42)
pos_test = df_positive.drop(pos_train.index).drop(pos_val.index).sample(n=30, random_state=42)

neg_train = df_negative.sample(n=3000, random_state=42)
neg_val = df_negative.drop(neg_train.index).sample(n=500, random_state=42)
neg_test = df_negative.drop(neg_train.index).drop(neg_val.index).sample(n=500, random_state=42)

train_df = pd.concat([pos_train, neg_train]).sample(frac=1).reset_index(drop=True)
val_df = pd.concat([pos_val, neg_val]).sample(frac=1).reset_index(drop=True)
test_df = pd.concat([pos_test, neg_test]).sample(frac=1).reset_index(drop=True)

print('Train:', train_df.shape)
print('Validation:', val_df.shape)
print('Test:', test_df.shape)
```

## Phần 2: Tiền xử lý ảnh (Google Colab, images\_001)

```python
import cv2
import numpy as np
from tqdm import tqdm
import os

# Thư mục ảnh gốc và thư mục lưu file .npy
image_folder = '/content/drive/MyDrive/Đồ án chuyên ngành Trí tuệ nhân tạo/Dataset/ChestX-ray14/images'
output_folder = '/content/drive/MyDrive/Đồ án chuyên ngành Trí tuệ nhân tạo/Dataset/ChestX-ray14/processed_npy'
os.makedirs(output_folder, exist_ok=True)

# Lọc lại các dataframe chỉ giữ ảnh có thật
available_images = set(os.listdir(image_folder))
for df_ in [train_df, val_df, test_df]:
    df_.drop(df_[~df_['Image Index'].isin(available_images)].index, inplace=True)
    df_.reset_index(drop=True, inplace=True)

# Chuẩn hóa theo ImageNet
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[LỖI] Không đọc được ảnh: {image_path}")
        return None
    img = cv2.equalizeHist(img)
    img = cv2.medianBlur(img, 3)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.stack([img, img, img], axis=-1)
    for i in range(3):
        img[:, :, i] = (img[:, :, i] - imagenet_mean[i]) / imagenet_std[i]
    return img.astype(np.float32)

# Tiền xử lý cả 3 tập train/val/test
for subset_df, name in zip([train_df, val_df, test_df], ['train', 'val', 'test']):
    for idx, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc=f'Processing {name}'):
        image_name = row['Image Index']
        image_path = os.path.join(image_folder, image_name)
        processed_img = preprocess_image(image_path)
        if processed_img is not None:
            save_name = os.path.splitext(image_name)[0] + '.npy'
            save_path = os.path.join(output_folder, save_name)
            np.save(save_path, processed_img)

print("✅ Đã tiền xử lý toàn bộ ảnh và lưu dưới dạng .npy")
```

## Phần 3: Xây dựng và huấn luyện Model A (Google Colab)

### Bước 3.1: Tính class weights để xử lý mất cân bằng

```python
from sklearn.utils import class_weight
import numpy as np

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=train_df['Label'].values
)

class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print("Class weights:", class_weight_dict)
```

### Bước 3.2: Custom Generator đọc ảnh .npy

```python
import tensorflow as tf
import os

class NPYDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, batch_size, data_dir, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = []
        batch_labels = []

        for i in batch_indices:
            row = self.df.iloc[i]
            image_name = os.path.splitext(row['Image Index'])[0] + '.npy'
            image_path = os.path.join(self.data_dir, image_name)

            if not os.path.exists(image_path):
                continue

            image = np.load(image_path)
            label = row['Label']
            batch_images.append(image)
            batch_labels.append(label)

        return np.array(batch_images), np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
```

### Bước 3.3: Tạo train/val generator

```python
train_generator = NPYDataGenerator(
    dataframe=train_df,
    batch_size=32,
    data_dir='/content/drive/MyDrive/Đồ án chuyên ngành Trí tuệ nhân tạo/Dataset/ChestX-ray14/processed_npy/'
)

val_generator = NPYDataGenerator(
    dataframe=val_df,
    batch_size=32,
    data_dir='/content/drive/MyDrive/Đồ án chuyên ngành Trí tuệ nhân tạo/Dataset/ChestX-ray14/processed_npy/',
    shuffle=False
)
```

### Bước 3.4: Xây dựng mô hình DenseNet121

```python
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models, optimizers

base_model = DenseNet121(include_top=False, input_shape=(224, 224, 3), pooling='avg', weights='imagenet')
x = layers.Dense(1, activation='sigmoid')(base_model.output)
model = models.Model(inputs=base_model.input, outputs=x)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### Bước 3.5: Huấn luyện mô hình

```python
model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    class_weight=class_weight_dict
)
```

### Bước 3.6: Lưu mô hình A sau khi huấn luyện

```
model_path = '/content/drive/MyDrive/Đồ án chuyên ngành Trí tuệ nhân tạo/models/model_A_chestxray14.h5'

model.save(model_path)

print("✅ Đã lưu Model A tại:", model_path)
```

## Phần 4: Đánh giá mô hình trên tập kiểm tra (test set)

### Bước 4.1: Đánh giá chỉ số loss và accuracy

```python
# Tạo generator cho tập test
test_generator = NPYDataGenerator(
    dataframe=test_df,
    batch_size=32,
    data_dir='/content/drive/MyDrive/Đồ án chuyên ngành Trí tuệ nhân tạo/Dataset/ChestX-ray14/processed_npy/',
    shuffle=False
)

# Đánh giá mô hình
loss, accuracy = model.evaluate(test_generator)
print("✅ Test accuracy:", accuracy)
print("✅ Test loss:", loss)
```

### Bước 4.2: Vẽ biểu đồ Accuracy và Loss theo từng Epoch

```python
import matplotlib.pyplot as plt

# Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy theo từng Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss theo từng Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

## Phần 5: Dự đoán và trực quan hóa kết quả

### Bước 5.1: Dự đoán và vẽ Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Dự đoán nhãn từ mô hình
y_true = test_df['Label'].values
predictions = model.predict(test_generator)
y_pred = (predictions > 0.5).astype(int)

# Tạo và hiển thị confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
```

### Bước 5.2: Hiển thị ảnh dự đoán sai (3x3 lưới)

```python
import random

wrong_indices = np.where(y_true != y_pred.reshape(-1))[0]
sample_wrong = random.sample(list(wrong_indices), min(9, len(wrong_indices)))

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 12))
axes = axes.flatten()

for idx, i in enumerate(sample_wrong):
    img_name = test_df.iloc[i]['Image Index']
    npy_path = os.path.join('/content/drive/MyDrive/Đồ án chuyên ngành Trí tuệ nhân tạo/Dataset/ChestX-ray14/processed_npy/', img_name.replace('.png', '.npy'))
    img = np.load(npy_path)
    axes[idx].imshow(img[:, :, 0], cmap='gray')
    axes[idx].set_title(f"Pred: {y_pred[i][0]}, True: {y_true[i]}")
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
```

## Phần 6: Chuẩn bị dữ liệu JSRT

### Bước 6.1: Gán nhãn từ jsrt\_metadata.csv

````python
import pandas as pd

# Đọc file metadata của JSRT
label_path = '/content/drive/MyDrive/JSRT/jsrt_metadata.csv'
df_jsrt = pd.read_csv(label_path)

# Gắn tên file ảnh tương ứng (.png)
df_jsrt['Image Index'] = df_jsrt['study_id'].apply(lambda x: x + '.png')

# Gán nhãn: 1 nếu malignant, ngược lại là 0 (cẩn thận với NaN)
df_jsrt['Label'] = df_jsrt['diagnosis'].apply(lambda x: 1 if isinstance(x, str) and x.lower() == 'malignant' else 0)

# In thống kê
print('✅ Số ảnh ung thư (label=1):', df_jsrt[df_jsrt['Label'] == 1].shape[0])
print('✅ Số ảnh không ung thư (label=0):', df_jsrt[df_jsrt['Label'] == 0].shape[0])
```python
import pandas as pd

# Đọc file metadata của JSRT
label_path = '/content/drive/MyDrive/JSRT/jsrt_metadata.csv'
df_jsrt = pd.read_csv(label_path)

# Gắn tên file ảnh tương ứng (.png)
df_jsrt['Image Index'] = df_jsrt['study_id'].apply(lambda x: x + '.png')

# Gán nhãn: 1 nếu malignant, ngược lại là 0
df_jsrt['Label'] = df_jsrt['diagnosis'].apply(lambda x: 1 if x.lower() == 'malignant' else 0)

# In thống kê
print('✅ Số ảnh ung thư (label=1):', df_jsrt[df_jsrt['Label'] == 1].shape[0])
print('✅ Số ảnh không ung thư (label=0):', df_jsrt[df_jsrt['Label'] == 0].shape[0])
````

### Bước 6.2: Tiền xử lý ảnh (giống như ChestX-ray14)

```python
import os
import cv2
import numpy as np
from tqdm import tqdm

input_folder = '/content/drive/MyDrive/JSRT/images'
output_folder = '/content/drive/MyDrive/JSRT/processed_npy'
os.makedirs(output_folder, exist_ok=True)

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[LỖI] Không đọc được ảnh: {image_path}")
        return None
    img = cv2.equalizeHist(img)
    img = cv2.medianBlur(img, 3)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.stack([img, img, img], axis=-1)
    for i in range(3):
        img[:, :, i] = (img[:, :, i] - imagenet_mean[i]) / imagenet_std[i]
    return img.astype(np.float32)

# Tiền xử lý toàn bộ JSRT
for idx, row in tqdm(df_jsrt.iterrows(), total=len(df_jsrt), desc='Processing JSRT'):
    img_name = row['Image Index']
    img_path = os.path.join(input_folder, img_name)
    processed = preprocess_image(img_path)
    if processed is not None:
        save_name = img_name.replace('.png', '.npy')
        save_path = os.path.join(output_folder, save_name)
        np.save(save_path, processed)

print("✅ Đã lưu toàn bộ ảnh JSRT dưới dạng .npy")
```

### Bước 6.3: Tính class weights để xử lý mất cân bằng

```python
from sklearn.utils import class_weight
import numpy as np

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=train_df['Label'].values
)

class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print("Class weights:", class_weight_dict)
```

## Phần 7: Huấn luyện Model B – từ đầu với JSRT



### Bước 7.1: Khởi tạo DenseNet121 từ ImageNet

\- Không dùng Model A

\- Add tầng đầu ra: \`Dense(1, activation='sigmoid')\`



### Bước 7.2: Áp dụng 10-fold Cross Validation

\- Mỗi fold huấn luyện riêng

\- Tổng hợp kết quả: accuracy, sensitivity, specificity



### Bước 7.3: Data Augmentation

\- Lật ảnh

\- Xoay ±30°



### Bước 7.4: Đánh giá mô hình

\- Confusion Matrix

\- Hiển thị ảnh dự đoán sai





## Phần 8: Huấn luyện Model C – fine-tune từ Model A



### Bước 8.1: Load Model A đã huấn luyện (ChestX-ray14)

\- \`model\_A\_chestxray14.h5\`



### Bước 8.2: Fine-tune với JSRT

\- Có thể freeze một phần hoặc toàn bộ base

\- Thay layer đầu ra nếu cần



### Bước 8.3: Huấn luyện + Augmentation

\- Cùng kỹ thuật như Model B

\- Có thể dùng lại folds đã chia ở Phần 6



### Bước 8.4: Đánh giá và so sánh với Model B

\- Confusion Matrix

\- Biểu đồ

\- So sánh kết quả giữa Model B và Model C





