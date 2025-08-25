import sys
import argparse
import logging
import os.path
import numpy as np
import joblib  
import pandas as pd
import keras
from keras import layers, callbacks, mixed_precision
import open_data
import model_creation
import model_evaluation
import model_history
import tensorflow as tf
import sklearn
from typing import Tuple

# Initialize data generators with default batch size
CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia", 
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax"
]
BATCH_SIZE = 8
train_gen, valid_gen, test_gen = open_data.get_generators(batch_size=BATCH_SIZE)
mixed_precision.set_global_policy('mixed_float16')

################################################################
#
# CNN Training Functions
#
def do_cnn_fit(my_args):
    #clear cache
    keras.backend.clear_session()
    if tf.config.list_physical_devices('GPU'):
        tf.config.experimental.reset_memory_stats('GPU:0')

    train_df, valid_df, _ = open_data.load_and_split_data()

    STEPS_PER_EPOCH = len(train_df) // BATCH_SIZE
    VALIDATION_STEPS = len(valid_df) // BATCH_SIZE

    train_gen, valid_gen, test_gen = open_data.get_generators(batch_size=BATCH_SIZE)
    train_gen = train_gen.repeat()

    def get_sample_labels(dataset, sample_batches=50):
        labels = []
        for X, y in dataset.take(sample_batches):
            labels.append(y.numpy())
        return np.concatenate(labels, axis=0)
    
    labels = get_sample_labels(train_gen)
    class_weights = compute_class_weights(labels)

    # 4. Memory callback (unchanged)
    class MemoryCleanupCallback(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            keras.backend.clear_session()
            if epoch % 2 == 0:
                tf.config.experimental.reset_memory_stats('GPU:0')
    
    weights_file = my_args.model_file.replace('.h5', '.weights.h5')

    # Callbacks-
    cb = [
        MemoryCleanupCallback(),
        callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc",
            patience=3,
            factor=0.5,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_pr_auc',
            patience=5,
            mode='max',
            restore_best_weights=True
        ),
        callbacks.ModelCheckpoint(
            filepath=weights_file,
            monitor='val_pr_auc',
            mode='max',
            save_best_only=True,
            save_weights_only=True
        ),
        callbacks.CSVLogger('training_log.csv')
    ]
    
    sample_X, _ = next(iter(train_gen))
    input_shape = sample_X.shape[1:]
    num_classes = labels.shape[-1]
    
    model = model_creation.create_model(
        my_args,
        input_shape,
        num_classes=num_classes
    )

    # 6. Train with adjusted steps
    history = model.fit(
        train_gen,
        steps_per_epoch=STEPS_PER_EPOCH,  # Reduced steps for safety
        validation_data=valid_gen,
        validation_steps=VALIDATION_STEPS,
        epochs=100,
        callbacks=cb,
        class_weight=class_weights,
        verbose=1
    )

    # 7. Post-training cleanup
    keras.backend.clear_session()
    model.load_weights(weights_file)  # Load best weights

    if not my_args.model_file.endswith('.h5'):
        my_args.model_file = f"{my_args.model_file}.h5"
    model.save(my_args.model_file)

    # Threshold optimization (with memory limits)
    optimal_thresholds = model_evaluation.optimize_thresholds(
        model=model,
        generator=valid_gen,
        num_classes=num_classes,
        max_batches=20  # Limit batches for threshold search
    )
    
    thresholds_file = my_args.model_file.replace('.h5', '_thresholds.npy')
    np.save(thresholds_file, optimal_thresholds)

    history_file = my_args.model_file.replace('.h5', '_history.joblib')
    joblib.dump(history.history, history_file)

    return history

def compute_class_weights(labels):
    pos_counts = np.sum(labels, axis=0)
    neg_counts = len(labels) - pos_counts
    total = pos_counts + neg_counts
    return {i: neg_counts[i]/pos_counts[i] if pos_counts[i] > 0 else 0 
            for i in range(len(pos_counts))}

def do_cnn_refit(my_args):
    """Refit an existing model with new data"""
    model_file = my_args.model_file
    model = joblib.load(model_file)
    
    cb = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        train_gen,
        epochs=10,
        validation_data=valid_gen,
        callbacks=cb,
        verbose=1
    )
    
    joblib.dump(model, model_file)
    joblib.dump(history.history, f"{model_file}.history")
    logging.info(f"Refitted model saved to {model_file}")
    return

################################################################
#
# Utility Functions
#
def cosine_annealing_with_warmup(epoch, warmup_epochs=5, max_lr=0.0003, 
                                min_lr=1e-6, total_epochs=50):
    """Cosine learning rate schedule with warmup"""
    if epoch < warmup_epochs:
        return (max_lr / warmup_epochs) * (epoch + 1)
    cos_inner = np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(cos_inner))

def parse_args(argv):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        prog=argv[0], 
        description='Image Classification with CNN'
    )
    
    parser.add_argument(
        'action',
        default='cnn-fit',
        choices=["cnn-fit", "score", "learning-curve", "cnn-refit"],
        nargs='?',
        help="desired action"
    )
    
    parser.add_argument(
        '--batch-number', '-b',
        default=1,
        type=int,
        help="training batch number (default=1)"
    )
    
    parser.add_argument(
        '--model-file', '-m',
        default="best_model.h5",
        type=str,
        help="model file path (default=model.joblib)"
    )
    
    parser.add_argument(
        '--model-name', '-M',
        default="a", 
        choices=["a","b"],
        type=str,
        help="model architecture name (default=a)"
    )
    
    return parser.parse_args(argv[1:])

def do_cnn_score(my_args):
    _, _, test_gen = open_data.get_generators(batch_size=32)
    model_evaluation.show_score(my_args, test_gen)

################################################################
#
# Main Execution
#
def main(argv):
    my_args = parse_args(argv)
    logging.basicConfig(level=logging.WARN)
    
    action_handlers = {
        'cnn-fit': do_cnn_fit,
        'cnn-refit': do_cnn_refit,
        'score': do_cnn_score,
        'learning-curve': model_history.plot_history
    }
    
    if my_args.action in action_handlers:
        action_handlers[my_args.action](my_args)
    else:
        raise ValueError(f"Unknown action: {my_args.action}")

if __name__ == "__main__":
    main(sys.argv)