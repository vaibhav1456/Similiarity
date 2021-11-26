from array import array
from itertools import chain
import logging
from math import sqrt

import numpy as np
from scipy import sparse

from gensim.matutils import corpus2csc
from gensim.utils import SaveLoad, is_corpus

logger = logging.getLogger(__name__)

NON_NEGATIVE_NORM_ASSERTION_MESSAGE = (
    u"sparse documents must not contain any explicit "
    u"zero entries and the similarity matrix S must satisfy x^T * S * x >= 0 for any "
    u"nonzero bag-of-words vector x."
)

class TermSimilarityIndex(SaveLoad):
    def most_similar(self, term, topn=10):
        raise NotImplementedError

    def __str__(self):
        members = ', '.join('%s=%s' % pair for pair in vars(self).items())
        return '%s<%s>' % (self.__class__.__name__, members)

class UniformTermSimilarityIndex(TermSimilarityIndex):
    def __init__(self, dictionary, term_similarity=0.5):
        self.dictionary = sorted(dictionary.items())
        self.term_similarity = term_similarity

    def most_similar(self, t1, topn=10):
        for __, (t2_index, t2) in zip(range(topn), (
                (t2_index, t2) for t2_index, t2 in self.dictionary if t2 != t1)):
            yield (t2, self.term_similarity)

class WordEmbeddingSimilarityIndex(TermSimilarityIndex):
    def __init__(self, keyedvectors, threshold=0.0, exponent=2.0, kwargs=None):
        self.keyedvectors = keyedvectors
        self.threshold = threshold
        self.exponent = exponent
        self.kwargs = kwargs or {}
        super(WordEmbeddingSimilarityIndex, self).__init__()
    def most_similar(self, t1, topn=10):
        if t1 not in self.keyedvectors:
            logger.debug('an out-of-dictionary term "%s"', t1)
        else:
            most_similar = self.keyedvectors.most_similar(positive=[t1], topn=topn, **self.kwargs)
            for t2, similarity in most_similar:
                if similarity > self.threshold:
                    yield (t2, similarity**self.exponent)

def _shortest_uint_dtype(max_value):
    if max_value < 2**8:
        return np.uint8
    elif max_value < 2**16:
        return np.uint16
    elif max_value < 2**32:
        return np.uint32
    return np.uint64

def _create_source(index, dictionary, tfidf, symmetric, dominant, nonzero_limit, dtype):
    assert isinstance(index, TermSimilarityIndex)
    assert dictionary is not None
    matrix_order = len(dictionary)
    if matrix_order == 0:
        raise ValueError('Dictionary provided to SparseTermSimilarityMatrix must not be empty')
    logger.info("constructing a sparse term similarity matrix using %s", index)
    if nonzero_limit is None:
        nonzero_limit = matrix_order

    def tfidf_sort_key(term_index):
        if isinstance(term_index, tuple):
            term_index, *_ = term_index
        term_idf = tfidf.idfs[term_index]
        return (-term_idf, term_index)
    if tfidf is None:
        columns = sorted(dictionary.keys())
        logger.info("iterating over %i columns in dictionary order", len(columns))
    else:
        assert max(tfidf.idfs) == matrix_order - 1
        columns = sorted(tfidf.idfs.keys(), key=tfidf_sort_key)
        logger.info("iterating over %i columns in tf-idf order", len(columns))
    nonzero_counter_dtype = _shortest_uint_dtype(nonzero_limit)
    column_nonzero = np.array([0] * matrix_order, dtype=nonzero_counter_dtype)
    if dominant:
        column_sum = np.zeros(matrix_order, dtype=dtype)
    if symmetric:
        assigned_cells = set()
    row_buffer = array('Q')
    column_buffer = array('Q')
    if dtype is np.float16 or dtype is np.float32:
        data_buffer = array('f')
    elif dtype is np.float64:
        data_buffer = array('d')
    else:
        raise ValueError('Dtype %s is unsupported, use numpy.float16, float32, or float64.' % dtype)

    def cell_full(t1_index, t2_index, similarity):
        if dominant and column_sum[t1_index] + abs(similarity) >= 1.0:
            return True  # after adding the similarity, the matrix would cease to be strongly diagonally dominant
        assert column_nonzero[t1_index] <= nonzero_limit
        if column_nonzero[t1_index] == nonzero_limit:
            return True  # after adding the similarity, the column would contain more than nonzero_limit elements
        if symmetric and (t1_index, t2_index) in assigned_cells:
            return True  # a similarity has already been assigned to this cell
        return False
    
    def populate_buffers(t1_index, t2_index, similarity):
        column_buffer.append(t1_index)
        row_buffer.append(t2_index)
        data_buffer.append(similarity)
        column_nonzero[t1_index] += 1
        if symmetric:
            assigned_cells.add((t1_index, t2_index))
        if dominant:
            column_sum[t1_index] += abs(similarity)
    try:
        from tqdm import tqdm as progress_bar
    except ImportError:
        def progress_bar(iterable):
            return iterable
    for column_number, t1_index in enumerate(progress_bar(columns)):
        column_buffer.append(column_number)
        row_buffer.append(column_number)
        data_buffer.append(1.0)
        if nonzero_limit <= 0:
            continue
        t1 = dictionary[t1_index]
        num_nonzero = column_nonzero[t1_index]
        num_rows = nonzero_limit - num_nonzero
        most_similar = [
            (dictionary.token2id[term], similarity)
            for term, similarity in index.most_similar(t1, topn=num_rows)
            if term in dictionary.token2id
        ] if num_rows > 0 else []
        if tfidf is None:
            rows = sorted(most_similar)
        else:
            rows = sorted(most_similar, key=tfidf_sort_key)
        for t2_index, similarity in rows:
            if cell_full(t1_index, t2_index, similarity):
                continue
            if not symmetric:
                populate_buffers(t1_index, t2_index, similarity)
            elif not cell_full(t2_index, t1_index, similarity):
                populate_buffers(t1_index, t2_index, similarity)
                populate_buffers(t2_index, t1_index, similarity)
    data_buffer = np.frombuffer(data_buffer, dtype=dtype)
    row_buffer = np.frombuffer(row_buffer, dtype=np.uint64)
    column_buffer = np.frombuffer(column_buffer, dtype=np.uint64)
    matrix = sparse.coo_matrix((data_buffer, (row_buffer, column_buffer)), shape=(matrix_order, matrix_order))
    logger.info(
        "constructed a sparse term similarity matrix with %0.06f%% density",
        100.0 * matrix.getnnz() / matrix_order**2,
    )
    return matrix

def _normalize_dense_vector(vector, matrix, normalization):
    if not normalization:
        return vector
    vector_norm = vector.T.dot(matrix).dot(vector)[0, 0]
    assert vector_norm >= 0.0, NON_NEGATIVE_NORM_ASSERTION_MESSAGE
    if normalization == 'maintain' and vector_norm > 0.0:
        vector_norm /= vector.T.dot(vector)  
    vector_norm = sqrt(vector_norm)
    normalized_vector = vector
    if vector_norm > 0.0:
        normalized_vector /= vector_norm
    return normalized_vector

def _normalize_dense_corpus(corpus, matrix, normalization):
    if not normalization:
        return corpus
    corpus_norm = np.multiply(corpus.T.dot(matrix), corpus.T).sum(axis=1).T
    assert corpus_norm.min() >= 0.0, NON_NEGATIVE_NORM_ASSERTION_MESSAGE
    if normalization == 'maintain':
        corpus_norm /= np.multiply(corpus.T, corpus.T).sum(axis=1).T
    corpus_norm = np.sqrt(corpus_norm)
    normalized_corpus = np.multiply(corpus, 1.0 / corpus_norm)
    normalized_corpus = np.nan_to_num(normalized_corpus)  # account for division by zero
    return normalized_corpus

def _normalize_sparse_corpus(corpus, matrix, normalization):
    def __init__(self, source, dictionary=None, tfidf=None, symmetric=True, dominant=False,
            nonzero_limit=100, dtype=np.float32):
        if not sparse.issparse(source):
            index = source
            args = (index, dictionary, tfidf, symmetric, dominant, nonzero_limit, dtype)
            source = _create_source(*args)
            assert sparse.issparse(source)
        self.matrix = source.tocsc()
    def inner_product(self, X, Y, normalized=(False, False)):
        if not X or not Y:
            return self.matrix.dtype.type(0.0)
        normalized_X, normalized_Y = normalized
        valid_normalized_values = (True, False, 'maintain')
        if normalized_X not in valid_normalized_values:
            raise ValueError('{} is not a valid value of normalize'.format(normalized_X))
        if normalized_Y not in valid_normalized_values:
            raise ValueError('{} is not a valid value of normalize'.format(normalized_Y))
        is_corpus_X, X = is_corpus(X)
        is_corpus_Y, Y = is_corpus(Y)
        if not is_corpus_X and not is_corpus_Y:
            X = dict(X)
            Y = dict(Y)
            word_indices = np.array(sorted(set(chain(X, Y))))
            dtype = self.matrix.dtype
            X = np.array([X[i] if i in X else 0 for i in word_indices], dtype=dtype)
            Y = np.array([Y[i] if i in Y else 0 for i in word_indices], dtype=dtype)
            matrix = self.matrix[word_indices[:, None], word_indices].todense()
            X = _normalize_dense_vector(X, matrix, normalized_X)
            Y = _normalize_dense_vector(Y, matrix, normalized_Y)
            result = X.T.dot(matrix).dot(Y)
            if normalized_X is True and normalized_Y is True:
                result = np.clip(result, -1.0, 1.0)
            return result[0, 0]
        elif not is_corpus_X or not is_corpus_Y:
            if is_corpus_X and not is_corpus_Y:
                X, Y = Y, X  # make Y the corpus
                is_corpus_X, is_corpus_Y = is_corpus_Y, is_corpus_X
                normalized_X, normalized_Y = normalized_Y, normalized_X
                transposed = True
            else:
                transposed = False
            dtype = self.matrix.dtype
            expanded_X = corpus2csc([X], num_terms=self.matrix.shape[0], dtype=dtype).T.dot(self.matrix)
            word_indices = np.array(sorted(expanded_X.nonzero()[1]))
            del expanded_X
            X = dict(X)
            X = np.array([X[i] if i in X else 0 for i in word_indices], dtype=dtype)
            Y = corpus2csc(Y, num_terms=self.matrix.shape[0], dtype=dtype)[word_indices, :].todense()
            matrix = self.matrix[word_indices[:, None], word_indices].todense()
            X = _normalize_dense_vector(X, matrix, normalized_X)
            Y = _normalize_dense_corpus(Y, matrix, normalized_Y)
            result = X.dot(matrix).dot(Y)
            if normalized_X is True and normalized_Y is True:
                result = np.clip(result, -1.0, 1.0)
            if transposed:
                result = result.T
            return result
        else:  # if is_corpus_X and is_corpus_Y:
            dtype = self.matrix.dtype
            X = corpus2csc(X if is_corpus_X else [X], num_terms=self.matrix.shape[0], dtype=dtype)
            Y = corpus2csc(Y if is_corpus_Y else [Y], num_terms=self.matrix.shape[0], dtype=dtype)
            matrix = self.matrix
            X = _normalize_sparse_corpus(X, matrix, normalized_X)
            Y = _normalize_sparse_corpus(Y, matrix, normalized_Y)
            result = X.T.dot(matrix).dot(Y)
            if normalized_X is True and normalized_Y is True:
                result.data = np.clip(result.data, -1.0, 1.0)
            return result
