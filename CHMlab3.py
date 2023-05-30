import math


def equation(x):
   return 2 * math.cos(x / 2) - math.exp(x) + 1


def bisection_method(a, b, tolerance):
   xl = a
   xp = b
   iterations = 0

   while abs(xp - xl) > tolerance:
       iterations += 1
       xx = (xl + xp) / 2.0
       print("Iteration {}: xl = {:.6f}, xx = {:.6f}, xp = {:.6f}".format(iterations, xl, xx, xp))

       if equation(xx) == 0:
           return xx, equation(xx)

       if equation(xx) * equation(xl) < 0:
           xp = xx
       else:
           xl = xx

   root = (xl + xp) / 2.0
   residual = equation(root)

   print("\nSolution with tolerance 1e-3:")
   print("Root: {:.6f}".format(root))
   print("Residual: {:.6f}".format(residual))

   # Checking conditions for using iterative method
   if abs(equation(root)) < 1:
       print("Conditions for using the iterative method are satisfied.")
       iterative_method(root, 1e-5)
       iterative_method(root, 1e-6)
       iterative_method(root, 1e-8)


def iterative_method(x, tolerance):
   iterations = 0

   while abs(equation(x)) > tolerance:
       iterations += 1
       x = x - equation(x) / (-2 * math.sin(x / 2) - math.exp(x))
       residual = equation(x)
       print(
           "Iteration {}: Root: {:.8f}, Iterations: {}, Residual: {:.8f}".format(iterations, x, iterations, residual))

   print("\nSolution with tolerance 1e{}: {:.10f}".format(int(math.log10(tolerance)), x))


# Initial boundaries
a = 1
b = 2
tolerance = 1e-3

bisection_method(a, b, tolerance)