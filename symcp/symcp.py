"""Symbolic convex programming."""

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
        if self is other: # to make __hash__ work
            return True
        return EqualityConstraint(self, other)

    def __ge__(self, other):
        return InequalityConstraint(other, self)

    def __le__(self, other):
        return InequalityConstraint(self, other)

    # for numpy
    # def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    #     if method == '__call__':
    #         arr = inputs[0]
    #         assert isinstance(arr, np.ndarray)
    #         if ufunc is np.matmul:
    #             return self.T @ arr
    #         if ufunc is np.multiply:
    #             return self * arr
    #         if ufunc is np.add:
    #             return self + arr
    #         if ufunc is np.subtract:
    #             return -self + arr

    # basic numpy interop
    __array_priority__ = 1.

    # basic pandas interop
    __pandas_priority__ = 5000.

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

    def minimize(self, subject_to=()):
        """Minimize self subject to constraints.
        
        :returns: Program.
        :rtype: cp.Program
        """
        if len(self.shape) > 0:
            raise ValueError(
                f'Only scalars can be minimized, not shape {self.shape}')

    def _expression_recursive(self, **kwargs):

        for _, child in self.__dict__.items():
            if hasattr(child, "_expression_recursive"):
                child._expression_recursive(**kwargs)
        if hasattr(self, "_expression"):
            # pylint: disable=assignment-from-no-return
            self._current_expression = self._expression(**kwargs)
            return self.current_expression
        return None

    _current_expression = None

    def _expression(self, **kwargs):

        raise NotImplementedError

    @property
    def current_expression(self):
        """Current matrix expression."""
        return self._current_expression


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

    def __hash__(self):
        return id(self)

    def _expression(self, **kwargs):
        return {self: 1.},

    # def epigraph(self):
    #     """Epigraph transformation."""
    #     return Program(objective=self)

    def compile(self):
        return AffineExpression(linear={self:1.})

class EpigraphVariable(Variable):
    """Epigraph variable."""

class Parameter(Variable):
    """Symbolic parameter."""

    # for numpy
    def __array__(self):
        return self.value

    def _expression(self, **kwargs):
        return {}, self.value

    def compile(self):
        return AffineExpression(constant=self.value)

class Constant(Parameter):
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

    def _expression(self, **kwargs):

        le_va, le_co = self.left.current_expression
        re_va, re_co = self.right.current_expression
        constant = le_co + re_co
        variables_expr = {
            var: (le_va[var] if var in le_va else 0.)
                + (re_va[var] if var in re_va else 0.)
                for var in set(le_va).union(set(re_va))}

        return variables_expr, constant

    def compile(self):
        return self.left.compile() + self.right.compile()
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

    def compile(self):
        return Sum(self.left, -self.right).compile()

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

    @property
    def shape(self):
        return self.array.shape

    def compile(self):
        epigraph = EpigraphVariable(self.shape)
        return Program(
            linear= {epigraph:1.},
            constraints = ((self.array <= epigraph).compile(), (-self.array <= epigraph).compile()))

    def __le__(self, other):
        return Constraints(constraints=(self.array <= other, -self.array <= other))

class Neg(Transformation):
    """Negative of a symbolic array."""

    @property
    def value(self):
        """Value of the symbolic array."""
        return -self.array.value

    def compile(self):
        _ = self.array.compile()
        return AffineExpression(
            linear={k:-v for k,v in _.linear.items()}, constant=-_.constant)

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

class Constraints(Constraint):

    def __init__(self, constraints):
        self.constraints = constraints

    def __repr__(self):
        return ', '.join(str(c) for c in self.constraints)

    @property
    def value(self):
        """Is the constraint satisfied."""
        return all(c.value for c in self.constraints)

    def compile(self):
        return tuple(c.compile() for c in self.constraints)

class EqualityConstraint(Constraint):
    """Equality constraint."""

    @property
    def value(self):
        """Is the constraint satisfied."""
        return np.all(self.left.value == self.right.value)

    def __repr__(self):
        return str(self.left) + ' == ' + str(self.right)

    def compile(self):
        aff_expr = (self.left - self.right).compile()
        return LinearEqualityConstraint(linear=aff_expr.linear, constant=aff_expr.constant)

class InequalityConstraint(Constraint):
    """Inequality constraint."""

    @property
    def value(self):
        """Is the constraint satisfied."""
        return np.all(self.left.value <= self.right.value)

    def __repr__(self):
        return str(self.left) + ' <= ' + str(self.right)

    def compile(self):
        aff_expr = (self.left - self.right).compile()
        return LinearInequalityConstraint(linear=aff_expr.linear, constant=aff_expr.constant)


class AffineExpression:
    """Affine expression."""

    def __init__(
        self, linear = {}, constant = 0., ):
        self.linear = linear
        self.constant = constant

    @staticmethod
    def _add_dicts(d1, d2):
        return {
            k: (d1[k] if k in d1 else 0.) + (d2[k] if k in d2 else 0.)
            for k in set(d1).union(set(d2))}

    def __add__(self, other):
        if isinstance(other, AffineQuadraticExpression):
            return NotImplemented
        return AffineExpression(
            linear = self._add_dicts(self.linear, other.linear),
            constant = self.constant + other.constant)

    def __radd__(self, other):
        return self.__add__(other)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            + ', '.join([f'{k}={v}'for k, v in self.__dict__.items()])
            +")")

class AffineQuadraticExpression(AffineExpression):
    """Affine and quadratic expression."""

    def __init__(
        self, quadratic = {}, linear = {}, constant = 0., ):
        self.quadratic = quadratic
        super().__init__(linear=linear, constant=constant)

    def __add__(self, other):
        if isinstance(other, Program):
            return NotImplemented
        return AffineQuadraticExpression(
            quadratic = self._add_dicts(self.quadratic, other.quadratic) if hasattr(other, 'quadratic') else self.quadratic,
            linear = self._add_dicts(self.linear, other.linear),
            constant = self.constant + other.constant)
    

class Program(AffineQuadraticExpression):
    """Convex program."""
    
    def __init__(
        self, quadratic = {}, linear = {}, constant = 0., constraints = ()):
        super().__init__(quadratic=quadratic, linear=linear, constant=constant)
        self.constraints = constraints

    def __add__(self, other):
        return Program(
            quadratic = self._add_dicts(self.quadratic, other.quadratic) if hasattr(other, 'quadratic') else self.quadratic,
            linear = self._add_dicts(self.linear, other.linear),
            constant = self.constant + other.constant,
            constraints = tuple(
                list(self.constraints) + list(other.constraints)) if hasattr(other, 'constraints') else self.constraints )


class CompiledConstraint:
    """Compiled constraint."""

class LinearEqualityConstraint(AffineExpression):
    """Linear equality constraint"""

class LinearInequalityConstraint(AffineExpression):
    """Linear inequality constraint"""



def minimize(objective=0., constraints=()):
    """Minimize objective subject to constraints."""
    if not isinstance(objective, SymbolicArray):
        objective = Constant(objective)
    return objective.minimize(constraints=constraints)

# from dataclasses import dataclass

# @dataclass
# class Program:
#     quadratic: dict = {}
#     linear: dict = {}
#     constant: np.array = 0.
#     constraints: list = []

if __name__ == '__main__':

    x = Variable(3)

    c = Parameter(3)

    A = np.zeros((3,3))
    print(A @ x)

    import pandas as pd
    print(pd.Series(range(3)) * x)

    # x.T @ c