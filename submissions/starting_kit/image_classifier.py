import numpy as np

from skimage.transform import resize
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adam
from rampwf.workflows.image_classifier import get_nb_minibatches


class ImageClassifier(object):

    def __init__(self):
        inp = Input((32, 32, 3))
        x = Flatten(name='flatten')(inp)
        x = Dense(100, activation='relu', name='fc1')(x)
        out = Dense(403, activation='softmax', name='predictions')(x)
        self.model = Model(inp, out)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=1e-4),
            metrics=['accuracy'])
        self.batch_size = 16

    def _transform(self, x):
        # some images have a usually unused opacity channel which we
        # need to ignore
        if x.shape[2] == 4:
            x = x[:, :, 0:3]
        # cropping a middle square of the image
        h, w = x.shape[:2]
        min_shape = min(h, w)
        x = x[h // 2 - min_shape // 2:h // 2 + min_shape // 2,
              w // 2 - min_shape // 2:w // 2 + min_shape // 2]
        # resizing the image
        x = resize(x, (32, 32), preserve_range=True)
        # bringing input between 0 and 1
        x = x / 255.
        return x

    def _build_train_generator(self, img_loader, indices, batch_size,
                               shuffle=False):
        indices = indices.copy()
        nb = len(indices)
        X = np.zeros((batch_size, 32, 32, 3))
        Y = np.zeros((batch_size, 403))
        while True:
            if shuffle:
                np.random.shuffle(indices)
            for start in range(0, nb, batch_size):
                stop = min(start + batch_size, nb)
                # load the next minibatch in memory.
                # The size of the minibatch is (stop - start),
                # which is `batch_size` for the all except the last
                # minibatch, which can either be `batch_size` if
                # `nb` is a multiple of `batch_size`, or `nb % batch_size`.
                bs = stop - start
                Y[:] = 0
                for i, img_index in enumerate(indices[start:stop]):
                    x, y = img_loader.load(img_index)
                    x = self._transform(x)
                    X[i] = x
                    Y[i, y] = 1
                yield X[:bs], Y[:bs]

    def _build_test_generator(self, img_loader, batch_size):
        nb = len(img_loader)
        X = np.zeros((batch_size, 32, 32, 3))
        while True:
            for start in range(0, nb, batch_size):
                stop = min(start + batch_size, nb)
                # load the next minibatch in memory.
                # The size of the minibatch is (stop - start),
                # which is `batch_size` for the all except the last
                # minibatch, which can either be `batch_size` if
                # `nb` is a multiple of `batch_size`, or `nb % batch_size`.
                bs = stop - start
                for i, img_index in enumerate(range(start, stop)):
                    x = img_loader.load(img_index)
                    x = self._transform(x)
                    X[i] = x
                yield X[:bs]

    def fit(self, img_loader):
        np.random.seed(24)
        nb = len(img_loader)
        nb_train = int(nb * 0.9)
        nb_valid = nb - nb_train
        indices = np.arange(nb)
        np.random.shuffle(indices)
        ind_train = indices[0: nb_train]
        ind_valid = indices[nb_train:]

        gen_train = self._build_train_generator(
            img_loader,
            indices=ind_train,
            batch_size=self.batch_size,
            shuffle=True
        )
        gen_valid = self._build_train_generator(
            img_loader,
            indices=ind_valid,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.model.fit_generator(
            gen_train,
            steps_per_epoch=get_nb_minibatches(nb_train, self.batch_size),
            epochs=1,
            max_queue_size=16,
            workers=1,
            use_multiprocessing=True,
            validation_data=gen_valid,
            validation_steps=get_nb_minibatches(nb_valid, self.batch_size),
            verbose=1
        )

    def predict_proba(self, img_loader):
        nb_test = len(img_loader)
        gen_test = self._build_test_generator(img_loader, self.batch_size)
        return self.model.predict_generator(
            gen_test,
            steps=get_nb_minibatches(nb_test, self.batch_size),
            max_queue_size=16,
            workers=1,
            use_multiprocessing=True,
            verbose=0
        )
