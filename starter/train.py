import os
import argparse
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

logging.basicConfig(level=logging.INFO)

def train(model, train_loader, epochs):
    model.fit(train_loader, epochs=epochs, verbose=1)


def test(model, test_loader):
    loss, accuracy = model.evaluate(test_loader, verbose=1)
    logging.info(f'Test accuracy: {accuracy}')
    return accuracy

def create_model():
    base_model = EfficientNetB7(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    model = Sequential([
        base_model,
        Flatten(),
        Dense(5, activation='softmax')
    ])
    return model

def create_data_loaders(data_dir, batch_size):
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse',
        subset='training'
    )
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation'
    )
    return train_generator, validation_generator

def main(args):
    model = create_model()
    model.compile(optimizer=Adam(learning_rate=args.learning_rate),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[SparseCategoricalAccuracy()])

    train_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)
    train(model, train_loader, args.epochs)
    accuracy = test(model, test_loader)
    model.save(os.path.join('/opt/ml/model', '1'), save_format='tf')




if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data_dir', type=str, default=os.environ["SM_CHANNEL_TRAIN"],
)

    args = parser.parse_args()

    main(args)
