import cmath
import math


class Value:
    """stores a single scalar value and its gradient"""

    __slots__ = ("data", "grad", "requires_grad", "_prev", "_op", "_eml_x", "_eml_y")

    _CONST_CACHE = {}
    _GRAPH_CONST_CACHE = {}

    def __init__(self, data, _children=(), _op="", requires_grad=True):
        self.data = complex(data)
        self.grad = 0j
        self.requires_grad = requires_grad
        self._prev = tuple(_children)
        self._op = _op
        self._eml_x = None
        self._eml_y = None

    @classmethod
    def _const(cls, data):
        key = complex(data)
        cached = cls._CONST_CACHE.get(key)
        if cached is None:
            cached = cls(key, requires_grad=False)
            cls._CONST_CACHE[key] = cached
        return cached

    @classmethod
    def _graph_const(cls, key, build):
        cached = cls._GRAPH_CONST_CACHE.get(key)
        if cached is None:
            cached = build()
            cls._GRAPH_CONST_CACHE[key] = cached
        return cached

    @classmethod
    def _zero(cls):
        return cls._graph_const(
            "zero",
            lambda: cls._const(1).eml(
                cls._const(1).eml(cls._const(1)).eml(cls._const(1))
            ),
        )

    @classmethod
    def _log_zero(cls):
        return cls._graph_const(
            "log_zero",
            lambda: cls._const(1).eml(
                cls._const(1).eml(cls._zero()).eml(cls._const(1))
            ),
        )

    @classmethod
    def _log_half(cls):
        return cls._graph_const(
            "log_half",
            lambda: cls._const(1).eml(
                cls._const(1).eml(cls._const(0.5)).eml(cls._const(1))
            ),
        )

    @classmethod
    def _log_two(cls):
        return cls._graph_const(
            "log_two",
            lambda: cls._const(1).eml(
                cls._const(1).eml(cls._const(2)).eml(cls._const(1))
            ),
        )

    def eml(self, other):
        other = other if isinstance(other, Value) else self._const(other)
        if self.requires_grad:
            parents = (
                (self,) if self is other or not other.requires_grad else (self, other)
            )
        elif other.requires_grad:
            parents = (other,)
        else:
            parents = ()
        # Primitive from the paper: eml(x, y) = exp(x) - log(y).
        out = Value(
            cmath.exp(self.data) - self._log(other.data),
            parents,
            "eml",
            requires_grad=bool(parents),
        )
        out._eml_x = self
        out._eml_y = other
        return out

    def exp(self):
        # exp(x) = eml(x, 1)
        return self.eml(1)

    def log(self):
        # log(x) = eml(1, exp(eml(1, x)))
        one = self._const(1)
        return one.eml(one.eml(self).eml(one))

    def relu(self):
        # On the real axis: relu(x) = (x + sqrt(x^2)) / 2.
        # Every helper below is still written as an explicit EML lowering.
        one = self._const(1)
        # zero = log(1), cached because it is a constant-only EML subgraph.
        log_zero = self._log_zero()

        # square = exp(log(x) + log(x)) = x^2
        log_self = one.eml(one.eml(self).eml(one))
        neg_log_self = log_zero.eml(log_self.eml(one))
        add_logs = one.eml(one.eml(log_self).eml(one)).eml(neg_log_self.eml(one))
        square = add_logs.eml(one)

        # sqrt(square) = square^(1/2) = exp((1/2) * log(square))
        log_square = one.eml(one.eml(square).eml(one))
        log_half = self._log_half()
        log_log_square = one.eml(one.eml(log_square).eml(one))
        neg_log_log_square = log_zero.eml(log_log_square.eml(one))
        mul_logs = one.eml(one.eml(log_half).eml(one)).eml(neg_log_log_square.eml(one))
        sqrt = mul_logs.eml(one).eml(one)

        # numerator = x + sqrt(x^2)
        neg_sqrt = log_zero.eml(sqrt.eml(one))
        numerator = one.eml(one.eml(self).eml(one)).eml(neg_sqrt.eml(one))

        # relu(x) = numerator / 2 = exp(log(numerator) - log(2))
        log_numerator = one.eml(one.eml(numerator).eml(one))
        log_two = self._log_two()
        sub_logs = one.eml(one.eml(log_numerator).eml(one)).eml(log_two.eml(one))
        return sub_logs.eml(one)

    def __neg__(self):  # -self
        # -x = eml(log(0), exp(x))
        one = self._const(1)
        return self._log_zero().eml(self.eml(one))

    def __add__(self, other):
        other = other if isinstance(other, Value) else self._const(other)
        # x + y = eml(log(x), exp(-y))
        one = self._const(1)
        neg_other = self._log_zero().eml(other.eml(one))
        log_self = one.eml(one.eml(self).eml(one))
        return log_self.eml(neg_other.eml(one))

    def __radd__(self, other):  # other + self
        other = other if isinstance(other, Value) else self._const(other)
        # x + y = eml(log(x), exp(-y)), with operands swapped.
        one = self._const(1)
        neg_self = self._log_zero().eml(self.eml(one))
        log_other = one.eml(one.eml(other).eml(one))
        return log_other.eml(neg_self.eml(one))

    def __sub__(self, other):  # self - other
        other = other if isinstance(other, Value) else self._const(other)
        # x - y = eml(log(x), exp(y))
        one = self._const(1)
        log_self = one.eml(one.eml(self).eml(one))
        return log_self.eml(other.eml(one))

    def __rsub__(self, other):  # other - self
        other = other if isinstance(other, Value) else self._const(other)
        # x - y = eml(log(x), exp(y)), with operands swapped.
        one = self._const(1)
        log_other = one.eml(one.eml(other).eml(one))
        return log_other.eml(self.eml(one))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else self._const(other)
        # x * y = exp(log(x) + log(y))
        one = self._const(1)
        log_self = one.eml(one.eml(self).eml(one))
        log_other = one.eml(one.eml(other).eml(one))
        neg_log_other = self._log_zero().eml(log_other.eml(one))
        add_logs = one.eml(one.eml(log_self).eml(one)).eml(neg_log_other.eml(one))
        return add_logs.eml(one)

    def __rmul__(self, other):  # other * self
        other = other if isinstance(other, Value) else self._const(other)
        # x * y = exp(log(x) + log(y)), with operands swapped.
        one = self._const(1)
        log_other = one.eml(one.eml(other).eml(one))
        log_self = one.eml(one.eml(self).eml(one))
        neg_log_self = self._log_zero().eml(log_self.eml(one))
        add_logs = one.eml(one.eml(log_other).eml(one)).eml(neg_log_self.eml(one))
        return add_logs.eml(one)

    def __truediv__(self, other):  # self / other
        other = other if isinstance(other, Value) else self._const(other)
        # x / y = exp(log(x) - log(y))
        one = self._const(1)
        log_self = one.eml(one.eml(self).eml(one))
        log_other = one.eml(one.eml(other).eml(one))
        sub_logs = one.eml(one.eml(log_self).eml(one)).eml(log_other.eml(one))
        return sub_logs.eml(one)

    def __rtruediv__(self, other):  # other / self
        other = other if isinstance(other, Value) else self._const(other)
        # x / y = exp(log(x) - log(y)), with operands swapped.
        one = self._const(1)
        log_other = one.eml(one.eml(other).eml(one))
        log_self = one.eml(one.eml(self).eml(one))
        sub_logs = one.eml(one.eml(log_other).eml(one)).eml(log_self.eml(one))
        return sub_logs.eml(one)

    def __pow__(self, other):
        other = other if isinstance(other, Value) else self._const(other)
        # x^y = exp(y * log(x)) = exp(exp(log(y) + log(log(x))))
        one = self._const(1)
        log_self = one.eml(one.eml(self).eml(one))
        log_other = one.eml(one.eml(other).eml(one))
        log_log_self = one.eml(one.eml(log_self).eml(one))
        neg_log_log_self = self._log_zero().eml(log_log_self.eml(one))
        mul_inner = one.eml(one.eml(log_other).eml(one)).eml(neg_log_log_self.eml(one))
        return mul_inner.eml(one).eml(one)

    def backward(self):
        # topological order all of the children in the graph without recursion
        topo = []
        visited = set()
        stack = [(self, False)]

        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for child in node._prev:
                if child not in visited:
                    stack.append((child, False))

        # Each backward pass should start with fresh intermediate state.
        # Leaves keep accumulated grads, mirroring micrograd-style training loops.
        for node in topo:
            if node._prev:
                node.grad = 0j

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad += 1 + 0j
        for v in reversed(topo):
            if v._op != "eml":
                continue

            x = v._eml_x
            y = v._eml_y

            # d/dx eml(x, y) = exp(x), d/dy eml(x, y) = -1 / y.
            if x.requires_grad:
                x.grad += cmath.exp(x.data) * v.grad
            if y.requires_grad:
                if y.data == 0:
                    y.grad += complex(float("nan"), float("nan"))
                else:
                    y.grad += (-1 / y.data) * v.grad

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    @staticmethod
    def _log(data):
        if data == 0:
            real_sign = math.copysign(1.0, data.real)
            imag_sign = math.copysign(1.0, data.imag)
            if real_sign < 0:
                imag = math.copysign(math.pi, imag_sign)
            else:
                imag = math.copysign(0.0, imag_sign)
            return complex(float("-inf"), imag)
        return cmath.log(data)
