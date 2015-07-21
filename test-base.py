import numpy as np
from . import libsvm, liblinear
from . import libsparse

class BaseLibSVM(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base class for estimators that use libsvm as backing library
    This implements support vector machine classification and regression.
    Parameter documentation is in the derived `SVC` class.
    """

    # The order of these must match the integer values in LibSVM.
    # XXX These are actually the same in the dense case. Need to factor
    # this out.
    _sparse_kernels = ["linear", "poly", "rbf", "sigmoid", "precomputed"]

    @abstractmethod
    def __init__(self, impl, kernel, degree, gamma, coef0,
                 tol, C, nu, epsilon, shrinking, probability, cache_size,
                 class_weight, verbose, max_iter, random_state):

        if not impl in LIBSVM_IMPL:  # pragma: no cover
            raise ValueError("impl should be one of %s, %s was given" % (
                LIBSVM_IMPL, impl))

        self._impl = impl
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.nu = nu
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.probability = probability
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.random_state = random_state
