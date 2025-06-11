# train.py
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from dataset import Dataset

#from tensorflow.keras.models import load_model
#model = load_model('model_checkpoint_epoch_04.h5')  # ostatni zapisany
#model.fit(...)  # kontynuuj trening z większą liczbą epok

BASE_PATH = "C:/Users/cadmus/PycharmProjects/PSI/clean_dataset/"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 6

checkpoint_cb = ModelCheckpoint(
    filepath='model_checkpoint_epoch_{epoch:02d}.h5',
    save_freq='epoch',
    save_weights_only=False,
    save_best_only=False
)

def load_dataset_from_df(df):
    paths = df['path'].apply(lambda p: os.path.join(BASE_PATH, p)).tolist()
    labels = df['age'].tolist()

    def _parse_function(filename, label):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMG_SIZE)
        image = image / 255.0
        return image, tf.cast(label, tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset


def train_model():
    dataset = Dataset(BASE_PATH)
    dataset.load_dataset()
    df = dataset.df_clean.copy()

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = load_dataset_from_df(train_df)
    val_dataset = load_dataset_from_df(val_df)

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1)(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=[checkpoint_cb])

    model.save("model_age_estimation.h5")
    print("Model zapisany jako model_age_estimation.h5")
