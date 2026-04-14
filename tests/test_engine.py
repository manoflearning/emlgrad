from __future__ import annotations

import cmath
import math
import random
import unittest

from emlgrad import Value, nn


def finite_difference(fn, z: complex, step: complex) -> complex:
    return (fn(z + step) - fn(z - step)) / (2 * step)


def finite_difference_x(fn, x: complex, y: complex, step: complex) -> complex:
    return finite_difference(lambda z: fn(z, y), x, step)


def finite_difference_y(fn, x: complex, y: complex, step: complex) -> complex:
    return finite_difference(lambda z: fn(x, z), y, step)


class ValueTests(unittest.TestCase):
    def assertComplexAlmostEqual(
        self,
        actual: complex,
        expected: complex,
        *,
        places: int = 9,
    ) -> None:
        self.assertAlmostEqual(actual.real, expected.real, places=places)
        self.assertAlmostEqual(actual.imag, expected.imag, places=places)

    def assertGradientMatchesFiniteDifference(
        self,
        grad: complex,
        fn,
        z: complex,
        *,
        places: int = 5,
    ) -> None:
        self.assertComplexAlmostEqual(
            grad,
            finite_difference(fn, z, complex(1e-7, 0.0)),
            places=places,
        )
        self.assertComplexAlmostEqual(
            grad,
            finite_difference(fn, z, complex(0.0, 1e-7)),
            places=places,
        )

    def test_eml_forward_and_backward(self) -> None:
        x = Value(0.3 + 0.2j)
        y = Value(1.7 - 0.4j)

        out = x.eml(y)
        out.backward()

        self.assertComplexAlmostEqual(out.data, cmath.exp(x.data) - cmath.log(y.data))
        self.assertComplexAlmostEqual(x.grad, cmath.exp(x.data))
        self.assertComplexAlmostEqual(y.grad, -1 / y.data)

    def test_unary_forward_matches_python(self) -> None:
        values = [2.5, -2.5, 1.2 + 0.3j, -1.2 + 0.3j]
        cases = [
            ("neg", lambda x: (-x).data, lambda z: -z),
            ("exp", lambda x: x.exp().data, cmath.exp),
            ("log", lambda x: x.log().data, cmath.log),
        ]

        for name, actual_fn, expected_fn in cases:
            for value in values:
                with self.subTest(op=name, value=value):
                    self.assertComplexAlmostEqual(
                        actual_fn(Value(value)), expected_fn(complex(value))
                    )

    def test_binary_forward_matches_python(self) -> None:
        pairs = [
            (2.5, 0.75),
            (1.2 + 0.3j, 0.8 - 0.1j),
            (-2.5, 0.75),
        ]
        cases = [
            ("add", lambda x, y: (x + y).data, lambda a, b: a + b),
            ("sub", lambda x, y: (x - y).data, lambda a, b: a - b),
            ("mul", lambda x, y: (x * y).data, lambda a, b: a * b),
            ("div", lambda x, y: (x / y).data, lambda a, b: a / b),
            ("pow", lambda x, y: (x**y).data, lambda a, b: a**b),
        ]

        for name, actual_fn, expected_fn in cases:
            for a, b in pairs:
                with self.subTest(op=name, a=a, b=b):
                    self.assertComplexAlmostEqual(
                        actual_fn(Value(a), Value(b)),
                        expected_fn(complex(a), complex(b)),
                        places=7,
                    )

    def test_multiplication_by_zero_stays_finite(self) -> None:
        self.assertComplexAlmostEqual((Value(0) * Value(2.5)).data, 0j)
        self.assertComplexAlmostEqual((Value(2.5) * Value(0)).data, 0j)

    def test_reflected_forward_matches_python(self) -> None:
        values = [2.5, -2.5, 1.2 + 0.3j]
        cases = [
            ("radd", lambda x: (2 + x).data, lambda z: 2 + z),
            ("rsub", lambda x: (2 - x).data, lambda z: 2 - z),
            ("rmul", lambda x: (2 * x).data, lambda z: 2 * z),
            ("rtruediv", lambda x: (2 / x).data, lambda z: 2 / z),
        ]

        for name, actual_fn, expected_fn in cases:
            for value in values:
                with self.subTest(op=name, value=value):
                    self.assertComplexAlmostEqual(
                        actual_fn(Value(value)), expected_fn(complex(value)), places=7
                    )

    def test_log_matches_principal_branch_on_negative_real_axis(self) -> None:
        out = Value(-2.5).log()
        self.assertComplexAlmostEqual(out.data, cmath.log(-2.5 + 0j))

    def test_log_zero_returns_negative_infinity_real_part(self) -> None:
        out = Value(0).log()
        self.assertTrue(math.isinf(out.data.real))
        self.assertLess(out.data.real, 0.0)
        self.assertEqual(out.data.imag, 0.0)

    def test_unary_gradients_match_finite_differences(self) -> None:
        z = 1.3 + 0.2j
        cases = [
            ("neg", lambda x: -x, lambda t: -t),
            ("exp", lambda x: x.exp(), cmath.exp),
            ("log", lambda x: x.log(), cmath.log),
        ]

        for name, build, ref in cases:
            with self.subTest(op=name):
                x = Value(z)
                out = build(x)
                out.backward()
                self.assertGradientMatchesFiniteDifference(x.grad, ref, z)

    def test_binary_gradients_match_finite_differences(self) -> None:
        x0 = 1.3 + 0.2j
        y0 = 0.7 - 0.1j
        cases = [
            ("add", lambda x, y: x + y, lambda a, b: a + b),
            ("sub", lambda x, y: x - y, lambda a, b: a - b),
            ("mul", lambda x, y: x * y, lambda a, b: a * b),
            ("div", lambda x, y: x / y, lambda a, b: a / b),
            ("pow", lambda x, y: x**y, lambda a, b: a**b),
            ("eml", lambda x, y: x.eml(y), lambda a, b: cmath.exp(a) - cmath.log(b)),
        ]

        for name, build, ref in cases:
            with self.subTest(op=name):
                x = Value(x0)
                y = Value(y0)
                out = build(x, y)
                out.backward()
                self.assertComplexAlmostEqual(
                    x.grad,
                    finite_difference_x(ref, x0, y0, complex(1e-7, 0.0)),
                    places=5,
                )
                self.assertComplexAlmostEqual(
                    x.grad,
                    finite_difference_x(ref, x0, y0, complex(0.0, 1e-7)),
                    places=5,
                )
                self.assertComplexAlmostEqual(
                    y.grad,
                    finite_difference_y(ref, x0, y0, complex(1e-7, 0.0)),
                    places=5,
                )
                self.assertComplexAlmostEqual(
                    y.grad,
                    finite_difference_y(ref, x0, y0, complex(0.0, 1e-7)),
                    places=5,
                )

    def test_log_branch_cut_above_and_below_negative_real_axis(self) -> None:
        eps = 1e-6
        above = Value(-2 + 1j * eps).log().data
        below = Value(-2 - 1j * eps).log().data

        self.assertComplexAlmostEqual(above, cmath.log(-2 + 1j * eps), places=7)
        self.assertComplexAlmostEqual(below, cmath.log(-2 - 1j * eps), places=7)
        self.assertGreater(above.imag, 0.0)
        self.assertLess(below.imag, 0.0)

    def test_pow_branch_cut_above_and_below_negative_real_axis(self) -> None:
        eps = 1e-6
        exponent = Value(0.5)
        above = (Value(-2 + 1j * eps) ** exponent).data
        below = (Value(-2 - 1j * eps) ** exponent).data

        self.assertComplexAlmostEqual(above, (-2 + 1j * eps) ** 0.5, places=6)
        self.assertComplexAlmostEqual(below, (-2 - 1j * eps) ** 0.5, places=6)
        self.assertGreater(above.imag, 0.0)
        self.assertLess(below.imag, 0.0)

    def test_relu_matches_micrograd_on_real_inputs(self) -> None:
        cases = [
            (-2.0, 0.0, 0.0),
            (-0.5, 0.0, 0.0),
            (0.5, 0.5, 1.0),
            (3.0, 3.0, 1.0),
        ]

        for value, expected_forward, expected_grad in cases:
            with self.subTest(value=value):
                x = Value(value)
                out = x.relu()
                out.backward()
                self.assertAlmostEqual(out.data.real, expected_forward, places=7)
                self.assertAlmostEqual(x.grad.real, expected_grad, places=7)
                self.assertAlmostEqual(out.data.imag, 0.0, places=7)
                self.assertAlmostEqual(x.grad.imag, 0.0, places=7)

    def test_zero_grad_resets_all_parameters(self) -> None:
        random.seed(0)
        model = nn.MLP(2, [3, 1])
        out = model([Value(1.0), Value(-2.0)])
        loss = (out - 1) ** 2
        loss.backward()

        self.assertTrue(any(p.grad != 0 for p in model.parameters()))

        model.zero_grad()
        self.assertTrue(all(p.grad == 0 for p in model.parameters()))

    def test_regularization_stays_finite_with_random_biases(self) -> None:
        random.seed(0)
        model = nn.MLP(2, [3, 1])
        reg_loss = 1e-4 * sum((p * p for p in model.parameters()))

        self.assertTrue(math.isfinite(reg_loss.data.real))
        self.assertTrue(math.isfinite(reg_loss.data.imag))

    def test_mlp_one_step_training_reduces_loss(self) -> None:
        random.seed(0)
        model = nn.MLP(2, [4, 1])
        xs = [
            [Value(1.0), Value(-1.0)],
            [Value(-1.0), Value(1.0)],
            [Value(0.5), Value(0.5)],
        ]
        ys = [1.0, -1.0, 1.0]

        def loss():
            scores = [model(x) for x in xs]
            losses = [(1 + -y * score).relu() for y, score in zip(ys, scores)]
            return sum(losses) * (1.0 / len(losses))

        before = loss()
        model.zero_grad()
        before.backward()

        for p in model.parameters():
            p.data -= 0.01 * p.grad

        after = loss()
        self.assertLess(after.data.real, before.data.real)

    def test_backward_handles_deep_eml_chain_iteratively(self) -> None:
        x = Value(0.1)
        out = x
        for _ in range(3000):
            out = out * 1.0001

        out.backward()

        self.assertTrue(math.isfinite(x.grad.real))
        self.assertTrue(math.isfinite(x.grad.imag))


if __name__ == "__main__":
    unittest.main()
