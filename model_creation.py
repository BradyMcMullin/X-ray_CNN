
import keras
from keras import regularizers, activations, layers, metrics, models, saving
from keras import applications
from keras import ops


@saving.register_keras_serializable()
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-7  # Replaces ops.epsilon()
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return -ops.mean(alpha * ops.power(1. - pt, gamma) * ops.log(pt + epsilon))
    return focal_loss_fixed

def create_model(my_args, input_shape, num_classes):
    """
    Selects the correct function to build a model, based on the model name
    from the command line arguments.
    """
    create_functions = {
        "a": create_model_a,
        "b": create_model_b
    }
    if my_args.model_name not in create_functions:
        raise Exception("Invalid model name: {} not in {}".format(my_args.model_name, list(create_functions.keys())))
        
    model = create_functions[my_args.model_name](my_args, input_shape, num_classes=num_classes)
    print(model.summary())
    return model





def create_model_b(my_args, input_shape,num_classes=1):

    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))

    # Conv Block 1
    model.add(layers.Conv2D(64, (7,7),strides=2, padding='same',kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))


    model.add(layers.Conv2D(128, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation="relu"))
    model.add(layers.Conv2D(128, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.4))

    # Conv Block 2
    model.add(layers.Conv2D(256, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation="relu"))
    model.add(layers.Conv2D(256, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.4))

    # Conv Block 3
    model.add(layers.Conv2D(512, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation="relu"))
    model.add(layers.Conv2D(512, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation="relu"))
    model.add(layers.Conv2D(512, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.5))
    
    

    # Global Average Pooling
    model.add(keras.layers.GlobalAveragePooling2D())

    model.add(keras.layers.Dropout(0.5))

    #Dense and drop
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.Dropout(0.5))

    #Output Layer
    model.add(keras.layers.Dense(num_classes, activation="sigmoid"))

    # Compile model with ReduceLROnPlateau callback
    model.compile(
        loss="binary_crossentropy",
        metrics=["binary_accuracy",
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'), 
                metrics.AUC(name='auc', multi_label=True)],
        optimizer=keras.optimizers.AdamW(learning_rate=0.001, amsgrad=True )
    )

    return model

def create_model_a(my_args, input_shape,num_classes=1):
    base_model = applications.DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
    )

    for layer in base_model.layers[-20:]:
        layer.trainable = True

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    attention = layers.Dense(256, activation='relu')(x)
    attention = layers.Dense(1, activation='sigmoid')(attention)
    x = layers.multiply([x, attention])
    
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001, amsgrad=True, gradient_accumulation_steps=4),
        #optimizer=keras.optimizers.SGD(learning_rate=0.001, gradient_accumulation_steps=4),
        loss=focal_loss(),
        metrics=["binary_accuracy",
                metrics.AUC(name='auc', multi_label=True),
                keras.metrics.AUC(name='pr_auc', curve='PR')
            ],
        weighted_metrics=['precision', 'recall']
        
    )

    model.summary()
    return model


# ReduceLROnPlateau callback
