# 🖐️ TV Remote Gesture Recognition using Deep Learning

This project builds a gesture recognition system that classifies hand gestures mimicking TV remote control actions using deep learning. It uses a custom data generator and a 3D Convolutional Neural Network (Conv3D) to extract spatiotemporal features from gesture frame sequences.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Structure](#-dataset-structure)
- [Workflow](#-workflow)
- [Model Architecture](#-model-architecture)
- [Custom Data Generator](#-custom-data-generator)
- [Requirements](#-requirements)
- [How to Run](#-how-to-run)
- [Results](#-results)
- [Future Improvements](#-future-improvements)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## 🧠 Project Overview

- 🎯 **Goal**: Build a gesture recognition system to classify remote control gestures using image sequences.
- 📦 **Input**: Folder of gesture videos split into image frames.
- 📊 **Output**: Class predictions for each gesture.
- 🧠 **Deep Learning Stack**: 3D CNN using TensorFlow/Keras.

---

## 📂 Dataset Structure

The dataset is organized in folders by gesture class.

```
/Project_data/
├── train/
│   ├── gesture_0/
│   ├── gesture_1/
│   └── ...
└── val/
    ├── gesture_0/
    ├── gesture_1/
    └── ...
```

Each gesture folder contains multiple subfolders (each a gesture instance) with image frames.

---

## 🔁 Workflow

1. **Set seeds** for reproducibility (`numpy`, `random`, `tensorflow`)
2. **Load image frames** and map class labels
3. **Use a custom data generator** to yield batches of frame sequences
4. **Build a Conv3D model** to learn spatial and temporal patterns jointly
5. **Train the model** using the generator
6. **Evaluate model** on validation set using loss and accuracy metrics

---

## 🧠 Model Architecture

```python
model = Sequential()
model.add(Conv3D(32, (3, 3, 3), activation='relu', input_shape=(frame_count, height, width, 3)))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.25))

model.add(Conv3D(64, (3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
```

- 📦 Pure 3D CNN architecture
- 🎥 Handles spatial + temporal features directly via Conv3D
- 🔁 No RNNs (GRU/LSTM) are used

---

## 🔄 Custom Data Generator

To handle image sequences efficiently, a custom generator is used for feeding batches during training:

```python
def generator(source_path, folder_list, batch_size):
    while True:
        batch_data = np.zeros((batch_size, frame_count, height, width, 3))
        batch_labels = np.zeros((batch_size, num_classes))
        for i in range(batch_size):
            folder = random.choice(folder_list)
            imgs = sorted(os.listdir(os.path.join(source_path, folder)))[:frame_count]
            for j, img_name in enumerate(imgs):
                img = Image.open(os.path.join(source_path, folder, img_name)).resize((width, height))
                batch_data[i, j] = np.array(img) / 255.0
            label = get_label_from_folder_name(folder)  # You define this function
            batch_labels[i] = to_categorical(label, num_classes=num_classes)
        yield batch_data, batch_labels
```

---

## 📦 Requirements

```bash
pip install tensorflow numpy pandas pillow
```

---

## ▶️ How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/Shivani3797/LSTM_Gesture_Recognition.git
   cd tv-gesture-recognition
   ```

2. Make sure your dataset path is updated in the notebook:
   ```python
   base_dir = '/home/datasets/Project_data'
   ```

3. Launch and run the notebook:
   ```bash
   jupyter notebook TV_gesture.ipynb
   ```

---

## 📊 Results

- Stable training using data generator
- Efficient use of 3D convolutions for short gesture clips

---

## 🔮 Future Improvements

- Integrate OpenCV for real-time webcam-based gesture recognition
- Add data augmentation for generalization
- Explore hybrid architectures: Conv3D + Transformers

---

## 🙌 Acknowledgments

- TensorFlow/Keras for model training
- Gesture recognition ideas from smart home systems
- Community inspiration for data generator and architecture

---

## 📄 License

This project is licensed under the MIT License.
```
