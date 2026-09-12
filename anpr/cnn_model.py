from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten


class CNN_Model:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        model.add(
            Conv2D(
                32,
                (3, 3),
                padding="same",
                activation="relu",
                input_shape=(28, 28, 1),
            )
        )
        model.add(Conv2D(32, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(
            Conv2D(
                64,
                (3, 3),
                padding="same",
                activation="relu",
            )
        )
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(
            Conv2D(
                64,
                (3, 3),
                padding="same",
                activation="relu",
            )
        )
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation="softmax"))

        return model