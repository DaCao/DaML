import numpy as np




def _check_partial_fit_first_call(clf, classes=None):
    """
    This function returns True if it detects that this was the first call to
    ``partial_fit`` on ``clf``. In that case the ``classes_`` attribute is also
    set on ``clf``.
    """

    if getattr(clf, "classes_", None) is None and classes is None:
        raise ValueError("classes must be passed on the first call to _partial_fit. ")

    elif classes is not None:

        if getattr(clf, "classes_", None) is None:
            # this is the first call to partial_fit
            clf.classes_ = classes
            return True

        else:
            if not np.array_equal(clf.classes_, classes):
                raise ValueError(
                    "`classes=%r` is not the same as on last call "
                    "to partial_fit, was: %r" % (classes, clf.classes_))

    else:
        # classes is None but clf.classes) has been set so nothing to do
        pass

    return False






def logsumexp(X, axis):
    """
    With a trick, Computes the logarithm of the sum of exponentials (a common quantity in ML) of input elements
    along given axis.
    References:
    https://stats.stackexchange.com/questions/105602/example-of-how-the-log-sum-exp-trick-works-in-naive-bayes
    https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
    https://github.com/scipy/scipy/blob/v0.19.0/scipy/special/_logsumexp.py#L8-L127

    :param X: ndarray
    :param axis: integer
    :return: 1d-array of floats
    """
    # get max values along axis in X
    a_max = np.amax(X, axis=axis, keepdims=True)

    # change inf to 0
    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0

    tmp = np.exp(X - a_max)
    out = np.log(np.sum(tmp, axis=axis))
    out += np.squeeze(a_max, axis=axis)
    return out



if __name__ == "__main__":

    x = np.random.rand(10, 6)
    a_max = np.amax(x, axis=1, keepdims=True)
    print(x)
    print(a_max)
    print(a_max.shape)
    print('\n-------------\n')
    print(np.squeeze(a_max, axis=1))
    print('\n-------------\n')

    res = logsumexp(x,1)
    print(res)
    print('\n-------------\n')


    print(np.atleast_2d(res))
    print(np.atleast_2d(res).T)

    x - np.atleast_2d(res).T


    exit()

    print('\n-------------\n')
    x[2:8, 3:5] = np.inf
    print(x)
    print('\n-------------\n')
    print(~np.isfinite(x))
    print('\n-------------\n')
    x[~np.isfinite(x)] = 0
    print(x)
    print('\n-------------\n')
    print(x-a_max)







