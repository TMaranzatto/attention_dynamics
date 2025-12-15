from sympy import Symbol, symbols, Matrix, Rational, N, simplify, Float, conjugate, substitution
#matrices are generated here
v_syms  = symbols('v0:%d:%d' % (2,2))
a_syms = symbols('a:%d:%d' % (2,2))
V = Matrix(2,2, v_syms)
A = Matrix(2,2, a_syms)

#this is the placeholder for the 2nd order parameter, interpret it as such
r = symbols('r')


w0 = V[0, 0] * A[1, 0] + V[0, 1] * A[1, 1] - V[1, 0] * A[0, 0] - V[1, 1] * A[0, 1]
w1 = V[0, 0] * A[0, 0] + V[0, 1] * A[0, 1] - V[1, 0] * A[1, 0] - V[1, 1] * A[1, 1]
w2 = -V[0, 0] * A[1, 0] - V[0, 1] * A[1, 1] - V[1, 0] * A[0, 0] - V[1, 1] * A[0, 1]
w3 = V[0, 0] * A[1, 0] - V[0, 1] * A[1, 1] - V[1, 0] * A[0, 0] + V[1, 1] * A[0, 1]
w4 = V[0, 0] * A[1, 1] + V[0, 1] * A[1, 0] - V[1, 0] * A[0, 1] - V[1, 1] * A[0, 0]
w5 = V[0, 0] * A[0, 0] - V[0, 1] * A[0, 1] - V[1, 0] * A[1, 0] + V[1, 1] * A[1, 1]
w6 = V[0, 0] * A[0, 1] + V[0, 1] * A[0, 0] - V[1, 0] * A[1, 1] - V[1, 1] * A[1, 0]
w7 = V[0, 0] * A[1, 0] - V[0, 1] * A[1, 1] + V[1, 0] * A[0, 0] - V[1, 1] * A[0, 1]
w8 = V[0, 0] * A[1, 1] + V[0, 1] * A[1, 0] + V[1, 0] * A[0, 1] + V[1, 1] * A[0, 0]

b1 = 1 / 16 * (1j * w5 + w6 + w7 - 1j * w8)
b2 = 1 / 16 * (1j * w5 - w6 + w7 + 1j * w8)
b3 = 1 / 8 * (1j * w1 - w2)

c1 = 1 / 8 * (-w3 + 1j * w4)
c2 = - 1 / 4 * w0

B = b1*r + b2 * conjugate(r) + b3
C = c1*r + conjugate(c1 * r) + c2

H = 2j*conjugate(B)
Omega = C

a = Symbol("a", real=True)
#formatting this to be more indicative of the matrix structure
substitution_set = {V[0,0]: Symbol("A00", real=True), V[0,1]: a,
                    V[1,0]:a, V[1,1]:Symbol("A11", real=True),

                    A[0, 0]: 1, A[0, 1]: 0,
                    A[1, 0]: 0, A[1, 1]: 1}

print(f"H(t) = {H.subs(substitution_set)}")
print(f"Omega(t) = {Omega.subs(substitution_set)}")

