from tensorflow.keras import models, layers


def build(dim):

    model = models.Sequential([

        layers.Dense(64, activation="relu", input_shape=(dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(dim)

    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model
