"""Fast convex programming."""

import numpy as np

class SymbolicArray(object):
    """Base class."""

    _shape = ()

    _value = None

    def __add__(self, other):
        return Sum(self, other)

    def __radd__(self, other):
        return Sum(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    def __matmul__(self, other):
        return MatMul(self, other)

    def __rmatmul__(self, other):
        return MatMul(other, self)

    def __abs__(self):
        return Abs(self)

    def __neg__(self):
        return Neg(self)

    def __eq__(self, other):
        return EqualityConstraint(self, other)

    def __ge__(self, other):
        return InequalityConstraint(other, self)

    def __le__(self, other):
        return InequalityConstraint(self, other)

    @property
    def T(self):
        """Transpose."""
        return Transpose(self)

    @property
    def shape(self):
        """Shape."""
        return self._shape

    def __repr__(self):
        result = self.__class__.__name__ + '('
        result += ', '.join(
            ['shape='+str(self.shape)] +
            [name + '=' + str(attr) for name, attr in self.__dict__.items()
                if isinstance(attr, SymbolicArray)])
        return result + ')'

    def minimize(self, constraints=()):
        """Minimize self subject to constraints."""
        if len(self.shape) > 0:
            raise ValueError(
                f'Only scalars can be minimized, not shape {self.shape}')


class Variable(SymbolicArray):
    """Optimization variable."""
    def __init__(self, shape=()):
        if isinstance(shape, int):
            self._shape = (shape,)
        elif hasattr(shape, '__iter__') and np.all(
                (isinstance(el, int) and el > 0) for el in shape):
            self._shape = tuple(shape)
        else:
            raise SyntaxError(
                'Only positive integers, or iterables of, are valid shapes.')
        self._value = np.zeros(shape=self.shape)

    @property
    def value(self):
        """Value of the symbolic array."""
        return self._value

    @value.setter
    def value(self, value):
        self._value[:] = value

    # def epigraph(self):
    #     """Epigraph transformation."""
    #     return Program(objective=self)

class Parameter(Variable):
    """Symbolic parameter."""

class Constant(Variable):
    """Constant scalar or array."""

    def __init__(self, value):
        self._value = np.array(value)
        # if hasattr(value, 'shape'):
        self._shape = self._value.shape
        # elif np.isscalar(value):
        #    pass
        #else:
        #    raise ValueError(
        #        "Only numpy-compatible arrays or scalars can be constants.")
        # self._value = value





# class Expression(SymbolicArray):
#    """Combination of variables, parameters, constants."""

class Combination(SymbolicArray):
    """Combination of two symbolic arrays with shape broadcasting."""

    def __init__(self, left, right):
        if not isinstance(left, SymbolicArray):
            left = Constant(left)
        if not isinstance(right, SymbolicArray):
            right = Constant(right)
        self.left = left
        self.right = right
        self._shape = self._broadcast_shapes(left, right)

    def _broadcast_shapes(self, left, right):
        """Numpy broadcasting of shapes.
        
        See https://numpy.org/doc/stable/user/basics.broadcasting.html .
        """
        lenshape = max(len(left.shape), len(right.shape))
        result = []
        for i in range(lenshape):
            try:
                le = left.shape[-1-i]
            except IndexError:
                le = 1
            try:
                ri = right.shape[-1-i]
            except IndexError:
                ri = 1
            if le == 1 or ri == 1 or le == ri:
                result.append(max(le, ri))
            else:
                raise ValueError(
                    f'Sizes {le} and {ri} are not compatible.')
        return tuple(result[::-1])

class Sum(Combination):
    """Sum with Numpy broadcasting of two symbolic arrays."""

    @property
    def value(self):
        """Value of the symbolic array."""
        return self.left.value + self.right.value

    # def epigraph(self):
    #     """Epigraph transformation."""
    #     lepi = self.left.epigraph()
    #     repi = self.right.epigraph()
    #     return lepi + repi

class Sub(Combination):
    """Subtract with Numpy broadcasting of two symbolic arrays."""

    @property
    def value(self):
        """Value of the symbolic array."""
        return self.left.value - self.right.value

class Mul(Combination):
    """Multiplication with Numpy broadcasting of two symbolic arrays."""

    @property
    def value(self):
        """Value of the symbolic array."""
        return self.left.value * self.right.value

class MatMul(Combination):
    """Matrix multiplication with Numpy broadcasting of two symbolic arrays."""

    def _broadcast_shapes(self, left, right):
        # TODO: do this right
        return (left.value @ right.value).shape
    
    @property
    def value(self):
        """Value of the symbolic array."""
        return self.left.value @ self.right.value

class Transformation(SymbolicArray):
    """Transformation of a single symbolic array."""

    def __init__(self, array):
        if not isinstance(array, SymbolicArray):
            array = Constant(array)
        self.array = array
        self._shape = array.shape

class Abs(Transformation):
    """Absolute value elementwise of a symbolic array."""

    @property
    def value(self):
        """Value of the symbolic array."""
        return abs(self.array.value)

class Neg(Transformation):
    """Negative of a symbolic array."""

    @property
    def value(self):
        """Value of the symbolic array."""
        return -self.array.value

class Transpose(Transformation):
    """Transpose with Numpy rules of symbolic array.
    
    See https://numpy.org/doc/stable/reference/generated/numpy.transpose.html .
    """
    def __init__(self, array):
        super().__init__(array)
        self._shape = array.shape[::-1]

    @property
    def value(self):
        """Value of the symbolic array."""
        return np.transpose(self.array.value)

class Constraint(Combination):
    """Constraint."""

class EqualityConstraint(Constraint):
    """Equality constraint."""

    @property
    def value(self):
        """Is the constraint satisfied."""
        return np.all(self.left.value == self.right.value)

    def __repr__(self):
        return str(self.left) + ' == ' + str(self.right)

class InequalityConstraint(Constraint):
    """Inequality constraint."""

    @property
    def value(self):
        """Is the constraint satisfied."""
        return np.all(self.left.value <= self.right.value)

    def __repr__(self):
        return str(self.left) + ' <= ' + str(self.right)

class Program:
    """Convex program."""
    
    def __init__(self, objective = 0., constraints = ()):
        self.objective = objective
        self.constraints = constraints
    
    # def __add__(self, other):
    #     return Program(
    #         objective = self.objective + other.objective,
    #         constraints = tuple(
    #             list(self.constraints) + list(other.constraints)))

    def __repr__(self):
        return (
            f'Program(objective={self.objective},'
            f' constraints={self.constraints})')

def minimize(objective=0., constraints=()):
    """Minimize objective subject to constraints."""
    if not isinstance(objective, SymbolicArray):
        objective = Constant(objective)
    return objective.minimize(constraints=constraints)

            
if __name__ == '__main__':

    x = Variable(3)

    c = Parameter(3)

    # x.T @ c