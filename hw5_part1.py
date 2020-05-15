import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import hw5_part1_utils

from typing import Tuple
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import numpy as np

from tqdm import tqdm

# YOUR IMPLEMENTATION FOR THE SHADOW MODEL ATTACK GOES HERE ###################


def synthesize_attack_data(
    target_model: hw5_part1_utils.TargetModel,
    shadow_data: np.ndarray,
    shadow_labels: np.ndarray,
    num_shadow_models: int = 4
):
    """Synthesize attack data.

    Arguments:

        target_model {TargetModel} -- an instance of the TargetModel class;
          behaves as a keras model but additionally has a train_shadow_model
          function, which takes a subset of the shadow data and labels and
          returns a model with identical architecture and hyperparameters to
          the original target model, but that is trained on the given shadow
          data.

        shadow_data {np.ndarray} -- data available to the attack to train
          shadow models. If the arget model's training set is size N x D,
          shadow_data is 2N x D.

        shadow_labels {np.ndarray} -- the corresponding labels to the
          shadow_data, given as a numpy array of 2N integers in the range 0 to
          C where C is the number of classes.

        num_shadow_models {int} -- the number of shadow models to use when
          constructing the attack model's dataset.

    Returns: three np.ndarrays; let M = 2N * num_shadow_models

        attack_data {np.ndarray} [M, 2C] -- shadow data label probability and
           label one-hot

        attack_classes {np.ndarray} [M, 1 of {0,1,...,C-1}] -- shadow data
           labels

        attack_labels {np.ndarray} [M, 1 of {0,1}] -- attack data labels
           (training membership)

    """

    C = shadow_labels.max() + 1

    attack_data: np.ndarray = None
    attack_classes: np.ndarray = None
    attack_labels: np.ndarray = None

    # SOLUTION
    # raise NotImplementedError('You need to implement this.')

    in_data = []
    out_data = []
    shadow_data_classes = []

    for i in tqdm(
        range(num_shadow_models),
        desc="training shadow models",
        unit="split"
    ):

        split = hw5_part1_utils.DataSplit(shadow_labels, seed=i)

        shadow_model = target_model.train_shadow_model(
            shadow_data[split.in_idx], shadow_labels[split.in_idx],
            # shadow_data[split.out_idx], shadow_labels[split.out_idx]
            # validation data
        )

        in_pred = shadow_model.predict(shadow_data[split.in_idx])
        in_onehot = to_categorical(
            shadow_labels[split.in_idx], C
        )
        in_data.append(np.concatenate(
            (in_pred, in_onehot),
            axis=1)
        )

        out_pred = shadow_model.predict(shadow_data[split.out_idx])
        out_onehot = to_categorical(
            shadow_labels[split.out_idx], C
        )
        out_data.append(np.concatenate(
            (out_pred, out_onehot),
            axis=1)
        )
        shadow_data_classes.append(shadow_labels[split.in_idx])
        shadow_data_classes.append(shadow_labels[split.out_idx])

    in_data = np.concatenate(in_data)
    out_data = np.concatenate(out_data)

    attack_data = np.concatenate((in_data, out_data))

    attack_labels = np.concatenate((
        np.ones(len(in_data)),
        np.zeros(len(out_data)))
    )

    attack_classes = np.concatenate(shadow_data_classes)

    ###

    return attack_data, attack_classes, attack_labels


def build_attack_models(
    target_model: hw5_part1_utils.TargetModel,
    shadow_data: np.ndarray,
    shadow_labels: np.ndarray,
    num_shadow_models: int = 4,
    batch_size=2048,
    epochs=32
):
    """Build attacker models.

    Arguments:

        target_model {TargetModel} -- an instance of the TargetModel class;
          behaves as a keras model but additionally has a train_shadow_model
          function, which takes a subset of the shadow data and labels and
          returns a model with identical architecture and hyperparameters to
          the original target model, but that is trained on the given shadow
          data.

        shadow_data {np.ndarray} -- data available to the attack to train
          shadow models. If the arget model's training set is size N x D,
          shadow_data is 2N x D.

        shadow_labels {np.ndarray} -- the corresponding labels to the
          shadow_data, given as a numpy array of 2N integers in the range 0 to
          C where C is the number of classes.

        num_shadow_models {int} -- the number of shadow models to use when
          constructing the attack model's dataset.

    Returns:

        {tuple} -- a tuple of C keras models, where the c^th model predicts the
        probability that an instance of class c was a training set member.

    """

    attack_data, attack_classes, attack_labels = \
        synthesize_attack_data(
            target_model,
            shadow_data,
            shadow_labels,
            num_shadow_models=4
        )

    # to return
    attack_models: Tuple[Model] = None

    C = shadow_labels.max() + 1

    # SOLUTION
    # raise NotImplementedError('You need to implement this.')

    # Define the attack model architecture.
    def get_attack_model_architecture():
        attack_x = Input((2 * C,))
        #attack_y = Dense(128 * C, activation='relu')(attack_x)
        #attack_y = Dense(32 * C, activation='relu')(attack_x)
        #attack_y = Dense(8 * C, activation='relu')(attack_x)
        #attack_y = Dense(1, activation='sigmoid')(attack_y)

        attack_y = Dense(4 * C, activation='relu')(attack_x)
        attack_y = Dense(1, activation='sigmoid')(attack_y)

        attack_model = Model(attack_x, attack_y)

        attack_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return attack_model

    # Train the attack model. We have one model per ground truth class.
    ret_models = []

    for c in tqdm(range(C), desc="training attack models", unit="class"):
        attack_model = get_attack_model_architecture()

        attack_model.fit(
            attack_data[attack_classes == c],
            attack_labels[attack_classes == c],
            batch_size=batch_size,
            verbose=0,
            epochs=epochs
        )

        ret_models.append(attack_model)

    attack_models = tuple(ret_models)

    ###

    return attack_models


def evaluate_membership(attack_models, y_pred, y):
    """Evaluate the attacker about the membership inference

    Arguments:

        attack_model {tuple} -- a tuple of C keras models, where C is the
          number of classes.

        y_pred {np.ndarray} -- an N x C numpy array with the predictions of the
          model on the N instances we are performing the inference attack on.

        y {np.ndarray} -- the true labels for each of the instances given as a
          numpy array of N integers.

    Returns:

        {np.ndarray} -- an array of N floats in the range [0,1] representing
          the estimated probability that each of the N given instances is a
          training set member.

    """

    # To return
    preds: np.ndarray = None

    # SOLUTION
    # raise NotImplementedError('You need to implement this.')
    attack_in = np.concatenate((y_pred, to_categorical(y)), axis=1)

    preds = np.zeros(y.shape)

    for c in tqdm(range(len(attack_models)),
                  desc="evaluating submodels",
                  unit="class"):

        preds[y == c] = attack_models[c].predict(attack_in[y == c])[0]

    ###

    return preds

# YOU DO NOT NEED TO MODIFY THE REST OF THIS CODE. ############################


if __name__ == '__main__':
    # Load the dataset.
    data = hw5_part1_utils.CIFARData()

    # Make a target model for the dataset.
    target_model = \
        hw5_part1_utils.CIFARModel(
            epochs=48,
            batch_size=2048,
            noload=True, # prevents loading an existing pre-trained target
                         # model
        ).init(
            data.train, data.labels_train,
            # data.test, data.labels_test # validation data
        )

    tqdm.write('Building attack model...')
    attack_models = build_attack_models(
        target_model,
        data.shadow,
        data.labels_shadow
    )

    tqdm.write('Evaluating attack model...')
    y_pred_in = target_model.predict(data.train)
    y_pred_out = target_model.predict(data.test)

    tqdm.write('  Train Accuracy: {:.4f}'.format(
        (y_pred_in.argmax(axis=1) == data.labels_train).mean()))
    tqdm.write('  Test Accuracy:  {:.4f}'.format(
        (y_pred_out.argmax(axis=1) == data.labels_test).mean()))

    in_preds = evaluate_membership(
        attack_models,
        y_pred_in,
        data.labels_train
    )
    out_preds = evaluate_membership(
        attack_models,
        y_pred_out,
        data.labels_test
    )

    wrongs_in = y_pred_in.argmax(axis=1) != data.labels_train
    wrongs_out = y_pred_out.argmax(axis=1) != data.labels_test

    true_positives = (in_preds > 0.5).mean()
    true_negatives = (out_preds < 0.5).mean()
    attack_acc = (true_positives + true_negatives) / 2.

    attack_precision = (in_preds > 0.5).sum() / (
        (in_preds > 0.5).sum() + (out_preds > 0.5).sum()
    )

    # Compare to a baseline that merely guesses correct classified instances
    # are in and incorrectly classified instances are out.
    baseline_true_positives = \
        (y_pred_in.argmax(axis=1) == data.labels_train).mean()
    baseline_true_negatives = \
        (y_pred_out.argmax(axis=1) != data.labels_test).mean()
    baseline_attack_acc = \
        (baseline_true_positives + baseline_true_negatives) / 2.

    baseline_precision = \
        (y_pred_in.argmax(axis=1) == data.labels_train).sum() / (
            (y_pred_in.argmax(axis=1) == data.labels_train).sum() +
            (y_pred_out.argmax(axis=1) == data.labels_test).sum()
        )

    tqdm.write(
      f"\nTrue positive rate: {true_positives:0.4f}, " +
      f"true negative rate: {true_negatives:0.4f}"
    )
    tqdm.write(
      f"Shadow Attack Accuracy: {attack_acc:0.4f}, precision: {attack_precision:0.4f} " +
      f"(baseline: {baseline_attack_acc:0.4f}, {baseline_precision:0.4f})"
    )
