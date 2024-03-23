Symbolic Convex Programming
===========================

*This is not ready to use.*

Symcp is an experimental library for symbolic manipulation of convex programs.
It attempts a rather ambitious goal: a pure-Python implementation of a simple
user-facing syntax for specifying convex programs, automating the translation
into sparse (or dense) matrix form to be handled by numerical solvers.

There are already many mature open-source projects that accomplish similar
goals, so this project might never be completed. However, it the opinion of the
author that it is possible to achieve the above with minimal complexity and
maintenance cost, and fast execution time thanks to Numpy and Scipy
vectorization. This would also include caching of as many matrix manipulations
as possible, which is the key to unlock fast sequential programming.

Installation
------------

.. code::

    pip install symcp

Usage
-----

*This is not working (yet).*

.. code:: python

    import symcp
    import numpy as np

    N = 100
    Sigma = np.random.randn(N,N)
    Sigma = Sigma.T @ Sigma
    mu = np.random.randn(N)

    w = symcp.Variable(N)

    objective = w.T @ mu - 1/2 * w.T @ Sigma @ w
    program = objective.maximize(
        subject_to = [
            np.sum(w) == 1,
            w >= 0,
        ]
    )

    program.solve(solver='OSQP')

    print(w.value)



