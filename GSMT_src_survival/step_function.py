import numpy as np
from sklearn.utils import check_consistent_length

class StepFunction:
	def __init__(self, x, y, a = 1., b = 0.):
		#check_consistent_length(x,y)
		"""
			Callable the step function
			f(z) = a * y_i + b
			x_i <= z < x_{i + 1}
			Parameters:
				- x: ndarray, (n_points, ), values on the x axis in ascending order.
				- y: ndarray, (n_points, ), corresponding values on the y axis.
				- a: float, constant to multiply by.
				- b: float, constant offset term.
		"""
		self.x = x
		self.y = y
		self.a = a
		self.b = b
		#print(type(x), type(y))

	def __call__(self, x):
		x = np.atleast_1d(x)
		if not np.isfinite(x).all():
			raise ValueError('x must be finite')
		if np.min(x) < self.x[0] or np.max(x) > self.x[-1]:
			raise ValueError('x must be within [%f, %f]' % (self.x[0], self.x[-1]))
		i = np.searchsorted(self.x, x, side = 'left')
		#print(i)
		not_exact = self.x[i] != x
		i[not_exact] -= 1
		value = self.a * self.y[i] + self.b
		if value.shape[0] == 1:
			return value[0]
		return value

	def __repr__(self):
		return "StepFunction(x = %r, y =%r, a=%r, b=%r" % (self.x, self.y, self.a, self.b)
