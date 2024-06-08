import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


# +--------------------------------------+
# +     Chuẩn hóa và gán nhã dữ liệu     +
# +--------------------------------------+

Image_Width = 128
Image_Height = 128
Image_Size = (Image_Width, Image_Height)
Image_Channels = 3

filenames = os.listdir("./dogs-vs-cats/train")
categories = []

for f_name in filenames:
    category = f_name.split('.')[0]
    if category == 'dog':
        categories.append(1)  # ảnh dog thêm số 1 vào mảng
    else:
        categories.append(0)  # ảnh cat thêm số 0 vào mảng

# tạo một bảng dữ liệu với 2 cột là filenames và category
df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})


# +-----------------------------------------------+
# +     Khởi tạo mạng nơ ron và các lớp Layer     +
# +-----------------------------------------------+


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(Image_Width, Image_Height, Image_Channels)),
    # lớp chuẩn hóa
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax'),
])


# +--------------------------------------------------+
# +     biên dịch và chuẩn bị dữ liệu huấn luyện     +
# +--------------------------------------------------+


'''
biên dịch mô hình với hàm mất mát categorical_crossentropy, bộ tối ưu hóa RMSprop, chỉ số đánh giá là accuracy
    loss: đo sự khác biệt giữa nhãn thực tế và nhãn dự đoán
    optimizer: cập nhật trọng số để giảm mất mát
    metrics: đo hiệu suất của mô hình trên tập huấn luyện và tập kiểm thử
'''
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
earlystop = tf.keras.callbacks.EarlyStopping(patience=10)
# giảm tốc độ học đi một nửa (tối thiểu là 0.00001) khi mô hình không cải thiện được
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1,
                                                               factor=0.5, min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]

'''
    chia tập dữ liệu
'''
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size = 20

train_datagen = ImageDataGenerator(rotation_range=15,  # góc quay [-15, 15]
                                   rescale=1./255,  # chia tỉ lệ các giá trị điểm ảnh về khoảng [0, 1]
                                   shear_range=0.1,  # cắt méo ảnh trong khoảng [-0.1, 0.1]
                                   zoom_range=0.2,  # phóng to ảnh trong khoảng [0.8, 1.2]
                                   horizontal_flip=True,  # cho phép lật ngang ảnh
                                   width_shift_range=0.1,  # cho phép dịch chiều rộng ảnh trong khoảng[-0.1, 0.1]
                                   height_shift_range=0.1  # cho phép dịch chiều cao ảnh trong khoảng[-0.1, 0.1]
                                   )

train_generator = train_datagen.flow_from_dataframe(
    train_df, "./dogs-vs-cats/train/",
    x_col='filename',  # cột chứa tệp ảnh
    y_col='category',  # cột chứa nhãn
    target_size=Image_Size,  # kích thước ảnh mong muốn
    class_mode='categorical',  # chế độ phân loại là categorical
    batch_size=batch_size  # kích thước lô ảnh
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, "./dogs-vs-cats/train/",
    x_col='filename',
    y_col='category',
    target_size=Image_Size,
    class_mode='categorical',
    batch_size=batch_size
)


# +--------------------------------------------------+
# +     Tiến hành huấn luyện và đánh giá mô hình     +
# +--------------------------------------------------+

# huấn luyện mô hình
epochs = 10
history = model.fit(
    train_generator,  # dữ liệu huấn luyện
    epochs=epochs,  # học 10 lần qua tập huấn luyện
    validation_data=validation_generator,  # dữ liệu kiểm thử
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)
# lưu mô hình vào file Model.h5
model.save("Model.h5")

print(history.history.keys())
# #  "Accuracy"
# plt.plot(history.history['accuracy'], 'b*-')
# plt.plot(history.history['val_accuracy'], 'ro-')
# plt.ylabel('Độ chính xác huấn luyện (%)', fontsize=15)
# plt.xlabel('Vòng lặp', fontsize=15)
# plt.grid()
# plt.legend(['Huấn luyện', 'Kiểm tra'], loc='lower right', fontsize=13)
# plt.show()
#
# # "Loss"
# plt.plot(history.history['loss'], 'b*-')
# plt.plot(history.history['val_loss'], 'ro-')
# plt.ylabel('Giá trị hàm mất mát (%)', fontsize=15)
# plt.xlabel('Vòng lặp', fontsize=15)
# plt.grid()
# plt.legend(['Huấn luyện', 'Kiểm tra'], loc='upper right', fontsize=13)
# plt.show()
