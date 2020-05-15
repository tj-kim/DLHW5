import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from tensorflow.data import Dataset
from tensorflow.errors import OutOfRangeError
from tensorflow.keras.layers import Input, Dense, Concatenate, Activation
from tensorflow.keras.models import Model

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tqdm import tqdm as _tqdm

def _proj(of, onto):
    return K.sum((of * onto)) * onto / K.sum((onto**2) + 1e-6)


def _adversarial_train(
    X, y, z,
    C, A,
    cls_loss,
    cls_schedule,
    adv_loss,
    adv_schedule,
    parity,
    bootstrap_epochs=0,
    epochs=32,
    batch_size=32,
    display_progress=False
):

    N = len(X)

    alpha = K.placeholder(ndim=0, dtype='float32')
    lr = K.placeholder(ndim=0, dtype='float32')

    x_batch, y_batch, z_batch = (
        Dataset.from_tensor_slices((X, y, z))
            .shuffle(N)
            .repeat(2*epochs+bootstrap_epochs)
            .batch(batch_size)
            .make_one_shot_iterator()
            .get_next()
    )

    adversary_gradients = K.gradients(
        adv_loss(x_batch, y_batch, z_batch),
        A.trainable_weights
    )

    adversary_updates = [
        K.update_sub(w, lr * dw)
        for w, dw in zip(A.trainable_weights, adversary_gradients)
    ]

    update_adversary = K.function(
          [lr],
          [tf.constant(0)],
          updates=adversary_updates
    )

    classifier_gradients = K.gradients(
        cls_loss(x_batch, y_batch),
        C.trainable_weights
    )

    a_c_gradients = K.gradients(
        adv_loss(x_batch, y_batch, z_batch),
        C.trainable_weights
    )

    classifier_updates = [
        K.update_sub(w, lr * (dCdw - alpha * dAdw - _proj(dCdw, dAdw)))
        for w, dCdw, dAdw in zip(
            C.trainable_weights,
            classifier_gradients,
            a_c_gradients
        )
    ]

    update_classifier = K.function(
          [alpha, lr],
          [tf.constant(0)],
          updates=classifier_updates
    )

    if not display_progress:
        tqdm = _tqdm

    else:
        from matplotlib import pyplot as plt
        import matplotlib.ticker as ticker
        from IPython.display import DisplayHandle
        from IPython.display import clear_output

        from tqdm.notebook import tqdm as _tqdm_notebook
        tqdm = _tqdm_notebook

        dh = DisplayHandle()
        dh.display("Graphs loading ...")

        _X = Input((X.shape[1],))
        _Y = Input((y.shape[1],))
        _Z = Input((z.shape[1],))

        _classifier_loss = cls_loss(_X, _Y)
        _dcdcs = K.gradients(_classifier_loss, C.trainable_weights)

        _classifier_gradients = K.sum(
            [K.sum(K.abs(dcdc))
             for dcdc in _dcdcs]
        )

        _adversary_loss = adv_loss(_X, _Y, _Z)
        _dadas = K.gradients(_adversary_loss, A.trainable_weights)
        _adversary_gradients = K.sum(
            [K.sum(K.abs(dada))
             for dada in _dadas]
        )

        _dadcs = K.gradients(_adversary_loss, C.trainable_weights)

        _a_c_gradients = K.sum(
            [K.sum(K.abs(
                alpha * dadc -
                _proj(dcdc, dadc))) for dcdc, dadc in zip(_dcdcs, _dadcs)])

        _total_gradients = _classifier_gradients + _a_c_gradients

        cls_loss_f = K.function([_X, _Y], [_classifier_loss])
        cls_grad_f = K.function([_X, _Y], [_classifier_gradients])

        adv_loss_f = K.function([_X, _Y, _Z], [_adversary_loss])
        adv_grad_f = K.function([_X, _Y, _Z], [_adversary_gradients])

        a_c_grad_f = K.function([alpha, _X, _Y, _Z], [_a_c_gradients])
        total_grad_f = K.function([alpha, _X, _Y, _Z], [_total_gradients])

        adv_loss = []
        cls_loss = []

        adv_grad = []
        cls_grad = []
        a_c_grad = []
        total_grad = []

        adv_acc = []
        cls_acc = []
        dm_abs = []
        dm_rel = []
        dm_abs_ideal = []
        dm_rel_ideal = []
        dm_g0 = []
        dm_g1 = []
        base_class = []
        base_adv = []

        pred_range = []
        adv_range = []

        cls_lrs = []
        cls_alphas = []
        adv_lrs = []

        cls_xs = []
        adv_xs = []

        baseline_accuracy = max(y.mean(), 1-y.mean())
        baseline_adversary_accuracy = max(z.mean(), 1.0-z.mean())

    progress = tqdm(
        desc="training",
        unit="epoch",
        total=epochs,
        leave=False
    )

    def update_display(
        t,
        cls_lr=None, cls_alpha=None,
        adv_lr=None
    ):
        if not display_progress:
            return

        y_pred = C.predict(X)

        if cls_lr is not None:
            cls_xs.append(t)
            cls_lrs.append(cls_lr)
            cls_alphas.append(cls_alpha)

            cls_loss.append(cls_loss_f([X, y])[0])
            cls_grad.append(cls_grad_f([X, y])[0])
            a_c_grad.append(a_c_grad_f([cls_alpha, X, y, z])[0])
            total_grad.append(total_grad_f([cls_alpha, X, y, z])[0])

            y_acc = ((y_pred > 0.5) == y).mean()

            cls_acc.append(y_acc)
            base_class.append(baseline_accuracy)

            _dm = parity(y_pred > 0.5, y, z)
            dm_abs_ideal.append(0.0)
            dm_rel_ideal.append(1.0)
            dm_abs.append(abs(_dm[0]-_dm[1]))
            dm_rel.append(min(_dm[0],_dm[1])/(max(0.0001, _dm[0], _dm[1])))
            dm_g0.append(_dm[0])
            dm_g1.append(_dm[1])

            pred_range.append(y_pred.max() - y_pred.min())

        if adv_lr is not None:
            adv_xs.append(t)

            adv_lrs.append(adv_lr)

            adv_loss.append(adv_loss_f([X, y, z])[0])
            adv_grad.append(adv_grad_f([X, y, z])[0])

            z_pred = A.predict(x=[y_pred, z])
            z_acc = ((z_pred > 0.5) * 1 == z).mean()
            adv_acc.append(z_acc)

            base_adv.append(baseline_adversary_accuracy)
            adv_range.append(z_pred.max() - z_pred.min())

        fig, axs = plt.subplots(5, 1, figsize=(15, 15))

        axs1t = axs[1].twinx()
        axs2t = axs[2].twinx()
        axs3t = axs[3].twinx()

        axs[0].plot(cls_xs, cls_acc, label="classifier", color='green')
        axs[0].plot(cls_xs, base_class, label="baseline classifier", ls=':', color='green')
        axs[0].plot(adv_xs, adv_acc, label="adversary", color='red')
        axs[0].plot(adv_xs, base_adv, label="baseline adversary", ls=':', color='red')

        axs[1].plot(cls_xs, dm_abs, label="absolute disparity", color='red')
        axs[1].plot(cls_xs, dm_abs_ideal, label="ideal absolute disparity", color='red', ls=':')
        axs1t.plot(cls_xs, dm_rel, label="relative disparity", color='green')
        axs1t.plot(cls_xs, dm_rel_ideal, label="ideal relative disparity", color='green', ls=':')
        axs[1].plot(cls_xs, dm_g0, label="male positive", color="black")
        axs[1].plot(cls_xs, dm_g1, label="female positive", color="black")

        axs[2].plot(cls_xs, cls_loss, label="classifier cls loss", color='green')
        axs[2].plot(adv_xs, adv_loss, label="adversary loss", color='red')
        axs2t.plot(cls_xs, pred_range, label="classifier range", color='green', ls=':')
        axs2t.plot(adv_xs, adv_range, label="adversary range", color='red', ls=':')

        axs[4].plot(cls_xs, cls_lrs, label="classifier lr", color='green', ls=":")
        axs[4].plot(cls_xs, cls_alphas, label="classifier alpha", color='green', ls="-")
        axs[4].plot(adv_xs, adv_lrs, label="adversary lr", color='red', ls=":")

        axs[3].plot(cls_xs, total_grad, label="classifier (total) gradients", color='green')
        axs[3].plot(cls_xs, cls_grad, label="classifier (cls) gradients", color='green', ls=':')
        axs[3].plot(cls_xs, a_c_grad, label="classifier (adv) gradients", color='green', ls='-.')

        axs3t.plot(adv_xs, adv_grad, label="adversary gradients", color='red')

        axs[0].set_title("prediction performance")
        axs[1].set_title("fairness characteristics")
        axs[2].set_title("loss characteristics")
        axs[3].set_title("learning characteristics")
        axs[4].set_title("learning parameters")

        axs[2].set_yscale("log", basey=2.0)
        axs[3].set_yscale("symlog", basey=2.0)
        axs[4].set_yscale("symlog", basey=2.0)

        axs[0].set_ylabel("accuracy")

        axs[1].set_ylabel("outcome Pr")
        axs1t.set_ylabel("outcome (min group Pr) / (max group Pr)")
        axs[2].set_ylabel("loss")
        axs2t.set_ylabel("Pr or Pr diff")
        axs[3].set_ylabel("grad")
        axs3t.set_ylabel("grad")
        axs[4].set_ylabel("parameter val")

        axs1t.legend(loc=4)
        axs2t.legend(loc=4)
        axs3t.legend(loc=4)

        for axi, ax in enumerate(axs):
            ax.set_xlabel("t")
            ax.legend(loc=3)
            ax.yaxis.grid(True, which='major')

        dh.update(fig)

        plt.close(fig)

    adv_lr = adv_schedule(1)
    cls_alpha, cls_lr = cls_schedule(1)

    t = -bootstrap_epochs+1

    for _ in tqdm(
        range(bootstrap_epochs),
        desc="bootstrapping classifier",
        unit="epoch",
        leave=False
    ):
        for _ in tqdm(
            range(N // batch_size),
            desc="classifier",
            unit="batch",
            leave=False
        ):
            update_classifier([
                0.0,
                adv_lr
            ])

        update_display(t, cls_lr=cls_lr, cls_alpha=0.0)
        t += 1

    while True:
        try:
            adv_lr = adv_schedule(t)
            cls_alpha, cls_lr = cls_schedule(t)

            for _ in tqdm(
                range(N // batch_size),
                desc="adversary",
                unit="batch",
                leave=False
            ):
                update_adversary([adv_lr])

            update_display(t, adv_lr=adv_lr)

            for _ in tqdm(
                range(N // batch_size),
                desc="classifier",
                unit="batch",
                leave=False
            ):
                update_classifier([
                    cls_alpha,
                    cls_lr
                ])

            update_display(t, cls_lr=cls_lr, cls_alpha=cls_alpha)

            t += 1
            progress.update(1)

        except OutOfRangeError:
            break


class AdversarialFairModel(object):
    def __init__(self, classifier):
        # YOU DO NOT NEED TO MODIFY THIS CODE.
        self.classifier = classifier

    def predict(self, X):
        # YOU DO NOT NEED TO MODIFY THIS CODE.
        return self.classifier.predict(X)

    def _get_adversary_architecture(self):
        # raise NotImplementedError('You need to implement this.')

        # SOLUTION

        in_cls_pred = Input((1,))
        in_y = Input((1,))

        inv_in_cls_pred = \
            tf.log(in_cls_pred+0.00001) - tf.log(1.00001-in_cls_pred)

        ins = Concatenate()([inv_in_cls_pred, in_y])

        out = Dense(1, activation='sigmoid')(ins)

        return Model(inputs=[in_cls_pred, in_y], outputs=out)

    def train_dem_parity(
        self,
        X, y, z,
        epochs=32,
        batch_size=1024,
        display_progress=False
    ):

        # raise NotImplementedError('You need to implement this.')

        # SOLUTION

        N = len(X)
        C = self.classifier
        A = self._get_adversary_architecture()

        def adversary_loss(x, y, z):
            return K.sum(
                K.binary_crossentropy(
                    z,
                    A(inputs=[C(x), tf.zeros_like(y)])
                ),
                axis=0
            ) / N

        def classifier_loss(x, y):
            return K.sum(
                K.binary_crossentropy(y, C(x)), axis=0
            ) / N

        _adversarial_train(
            X, y, z,
            C, A,
            cls_loss = classifier_loss,
            cls_schedule = lambda t: (t**0.5, 10.0/t),
            adv_loss = adversary_loss,
            adv_schedule = lambda t: 10.0/t,
            parity=evaluate_dem_parity,
            bootstrap_epochs=4,
            epochs=epochs,
            batch_size=batch_size,
            display_progress=display_progress
        )

    def train_eq_op(
        self,
        X, y, z,
        epochs=32,
        batch_size=1024,
        display_progress=False
    ):
        # raise NotImplementedError('You need to implement this.')

        # SOLUTION

        N = len(X)
        C = self.classifier
        A = self._get_adversary_architecture()

        def adversary_loss(x, y, z):
            return K.sum(
                tf.boolean_mask(
                    K.binary_crossentropy(
                        z,
                        A(inputs=[C(x), y])
                    ),
                    tf.equal(y, 1.0)),
                    # Masked out negative ground truth here as otherwise we
                    # would be enforcing equalized odds. Another option is to
                    # train the adversary on only the portion of the dataset
                    # that has positive ground truth.
                axis=0
            ) / N

        def classifier_loss(x, y):
            return K.sum(
                K.binary_crossentropy(y, C(x)), axis=0
            ) / N

        _adversarial_train(
            X, y, z,
            C, A,
            cls_loss = classifier_loss,
            cls_schedule = lambda t: (t**0.5, 10.0/t),
            adv_loss = adversary_loss,
            adv_schedule = lambda t: 10.0/t,
            parity=evaluate_eq_op,
            bootstrap_epochs=4,
            epochs=epochs,
            batch_size=batch_size,
            display_progress=display_progress
        )


def evaluate_dem_parity(y_pred, y, z):
    # raise NotImplementedError('You need to implement this.')

    # SOLUTION ##
    return (
        (y_pred[z == 0] == 1).mean(),
        (y_pred[z == 1] == 1).mean()
    )


def evaluate_eq_op(y_pred, y, z):
    # raise NotImplementedError('You need to implement this.')

    # SOLUTION ##
    return (
        (y_pred[(z == 0) * (y == 1)] == 1).mean(),
        (y_pred[(z == 1) * (y == 1)] == 1).mean()
    )


def eval_result(label, model, X, y, z, baseline_accuracy=None):
    y_pred = (model.predict(X) > 0.5) * 1

    _tqdm.write(
        '{}\n'
        '------------------\n'
        '\n'
        'Accuracy: {:.4f} (baseline: {:0.4f})\n'
        '\n'
        '                    Group 0\tGroup 1\n'
        'Demographic Parity: {:.4f}\t{:.4f}\n'
        'Equal Opportunity:  {:.4f}\t{:.4f}\n'
        .format(label, *(
          ((y_pred == y).mean(), baseline_accuracy) +
          evaluate_dem_parity(y_pred, y, z) +
          evaluate_eq_op(y_pred, y, z))
        )
    )


def run_tests(display_progress):
    def norme(d):
        d = d.astype('float32')
        d -= d.min(axis=0)
        d -= d.max(axis=0) * 0.5
        d /= d.max(axis=0)
        return d

    def sample(d, g0, g1, n, balance=False):
        if balance:
            return np.vstack([d[g0][0:n//2], d[g1][0:n//2]])
        else:
            return d[0:n]

    temp = np.load("adult.npz")

    X, y, z = norme(temp['X']), \
        temp['y'].astype('float32'), \
        temp['z'].astype('float32')

    zi = z.reshape((len(z)))

    g0 = zi == 0.0
    g1 = zi == 1.0

    X, y, z = map(
        lambda d: sample(d, g0, g1, 30000, balance=False),
        (X, y, z)
    )

    baseline_accuracy = max(y.mean(), 1-y.mean())

    np.random.seed(0)
    tf.set_random_seed(0)

    def make_adult_classifier():
        c_in = Input((X.shape[1],))
        c_inter = Dense(32, activation="relu")(c_in)
        c_inter = Dense(32, activation="relu")(c_inter)
        c_inter = Dense(32, activation="relu")(c_inter)
        c_out = Dense(1, activation='sigmoid')(c_inter)

        return Model(c_in, c_out)

    if True:
        # Train model with demographic parity.
        c_dem_par = AdversarialFairModel(make_adult_classifier())
        c_dem_par.train_dem_parity(
            X, y, z, epochs=32, batch_size=16,
            display_progress=display_progress
        )

        eval_result(
            "Demographic Parity", c_dem_par,
            X, y, z, baseline_accuracy
        )

    if True:
        # Train model with equality of opportunity.
        c_eq_op = AdversarialFairModel(make_adult_classifier())
        c_eq_op.train_eq_op(
            X, y, z, epochs=32, batch_size=16,
            display_progress=display_progress
        )

        eval_result(
            "Equalized Opportunity", c_eq_op,
            X, y, z, baseline_accuracy
        )

    if True:
        # Train original model.
        c_orig = make_adult_classifier()
        c_orig.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        c_orig.fit(X, y, epochs=32, batch_size=16)

        eval_result(
            "Original Model", c_orig,
            X, y, z, baseline_accuracy
        )


if __name__ == '__main__':
    run_tests(display_progress=False)
