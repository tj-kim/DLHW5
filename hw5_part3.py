import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gensim.models
import numpy as np
from sklearn.decomposition import PCA
import json
from tqdm import tqdm


class WordEmbeddingDebiaser:

    def __init__(self,
                 embedding_file_path,
                 definitional_file_path='./data/definitional_pairs.json',
                 equalize_file_path='./data/equalize_pairs.json',
                 gender_specific_file_path='./data/gender_specific_full.json'):

        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            embedding_file_path, binary=True)

        # collect first 300000 words
        self.words = sorted([w for w in self.model.vocab],
                            key=lambda w: self.model.vocab[w].index)[:300000]

        # all vectors in an array (same order as self.words)
        self.vecs = np.array([self.model[w] for w in self.words])
        tqdm.write('vectors loaded')
        # should take 2-5 min depending on your machine

        self.n, self.d = self.vecs.shape

        # word to index dictionary
        self.w2i = {w: i for i, w in enumerate(self.words)}

        # Some relevant words sets required for debiasing
        with open(definitional_file_path, "r") as f:
            self.definition_pairs = json.load(f)

        with open(equalize_file_path, "r") as f:
            self.equalize_pairs = json.load(f)

        with open(gender_specific_file_path, "r") as f:
            self.gender_specific_words = json.load(f)
        self._normalize()

    # Some potentially helpful functions, you don't have to use/implement them.
    def _normalize(self):
        """
        normalize self.vecs
        """
        self.vecs /= np.linalg.norm(self.vecs, axis=1)[:, np.newaxis]

    def _drop(self, u, v):
        """
        remove a direction v from u
        """
        return u - v * u.dot(v) / v.dot(v)

    def w2v(self, word):
        """
        for a word, return its corresponding vector
        """
        return self.vecs[self.w2i[word]]

    def debias(self):
        self.gender_direction = self.identify_gender_subspace()
        self.neutralize()
        self.equalize()

    def identify_gender_subspace(self):
        """Using self.definitional_pairs to identify a gender axis (1 dimensional).

          Output: a gender direction using definitonal pairs

        ****Note****

         no other unimported packages listed above are allowed, please use
         numpy.linalg.svd for PCA

        """
        matrix = []
        for a, b in self.definition_pairs:
            center = (self.w2v(a) + self.w2v(b)) / 2
            matrix.append(self.w2v(a) - center)
            matrix.append(self.w2v(b) - center)
        matrix = np.array(matrix)
        # pca = PCA(n_components=10)
        # pca.fit(matrix)
        # gender_direction = pca.components_[0]
        u, s, v = np.linalg.svd(matrix)
        gender_direction = v[0, :]

        return gender_direction

        # raise NotImplementedError('You need to implement this.')

    def neutralize(self):
        """Performing the neutralizing step: projecting all gender neurtal words away
        from the gender direction

        No output, please adjust self.vecs

        """
        specific_set = set(self.gender_specific_words)
        for i, w in enumerate(self.words):
            if w not in specific_set:
                self.vecs[i] = self._drop(self.vecs[i], self.gender_direction)
        self._normalize()
        # raise NotImplementedError('You need to implement this.')

    def equalize(self):
        """Performing the equalizing step: make sure all equalized pairs are
        equaldistant to the gender direction.

        No output, please adapt self.vecs

        """
        for (a, b) in self.equalize_pairs:
            if (a in self.w2i and b in self.w2i):
                y = self._drop((self.w2v(a) + self.w2v(b)) / 2,
                               self.gender_direction)
                z = np.sqrt(1 - np.linalg.norm(y)**2)
                if (self.w2v(a) - self.w2v(b)).dot(self.gender_direction) < 0:
                    z = -z
                self.vecs[self.w2i[a]] = z * self.gender_direction + y
                self.vecs[self.w2i[b]] = -z * self.gender_direction + y
        self._normalize()
        # raise NotImplementedError('You need to implement this.')

    def compute_analogy(self, w3, w1='woman', w2='man'):
        """input: w3, w1, w2, satifying the analogy w1: w2 :: w3 : w4

        output: w4(a word string) which is the solution to the analogy (w4 is
          constrained to be different from w1, w2 and w3)

        """
        diff = self.w2v(w2) - self.w2v(w1)
        vec = diff / np.linalg.norm(diff) + self.w2v(w3)
        vec = vec / np.linalg.norm(vec)
        if w3 == self.words[np.argsort(vec.dot(self.vecs.T))[-1]]:
            return self.words[np.argsort(vec.dot(self.vecs.T))[-2]]
        return self.words[np.argmax(vec.dot(self.vecs.T))]


if __name__ == '__main__':

    # Original Embedding

    we = WordEmbeddingDebiaser('./data/GoogleNews-vectors-negative300.bin')

    print('=' * 50)
    print('Original Embeddings')
    # she-he analogy evaluation
    w3s1 = [
        'her', 'herself', 'spokeswoman', 'daughter', 'mother', 'niece',
        'chairwoman', 'Mary', 'sister', 'actress'
    ]
    w3s2 = [
        'nurse', 'dancer', 'feminist', 'baking', 'volleyball', 'softball',
        'salon', 'blond', 'cute', 'beautiful'
    ]

    w4s1 = [we.compute_analogy(w3) for w3 in w3s1]
    w4s2 = [we.compute_analogy(w3) for w3 in w3s2]

    print('Appropriate Analogies')
    for w3, w4 in zip(w3s1, w4s1):
        print("'woman' is to '%s' as 'man' is to '%s'" % (w3, w4))

    print('Potentially Biased Analogies')
    for w3, w4 in zip(w3s2, w4s2):
        print("'woman' is to '%s' as 'man' is to '%s'" % (w3, w4))

    we.debias()

    print('=' * 50)
    print('Debiased  Embeddings')
    # she-he analogy evaluation
    w4s1 = [we.compute_analogy(w3) for w3 in w3s1]
    w4s2 = [we.compute_analogy(w3) for w3 in w3s2]

    print('Appropriate Analogies')
    for w3, w4 in zip(w3s1, w4s1):
        print("'woman' is to '%s' as 'man' is to '%s'" % (w3, w4))

    print('Potentially Biased Analogies')
    for w3, w4 in zip(w3s2, w4s2):
        print("'woman' is to '%s' as 'man' is to '%s'" % (w3, w4))
