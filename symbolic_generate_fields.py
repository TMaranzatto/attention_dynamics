from sympy import Symbol, symbols, Matrix, Rational, N, simplify, Float, conjugate, substitution, im, re, exp, Abs
#matrices are generated here
v_syms  = symbols('v0:%d:%d' % (2,2))
a_syms = symbols('a:%d:%d' % (2,2))
V = Matrix(2,2, v_syms)
A = Matrix(2,2, a_syms)

#this is the placeholder for the 2nd order parameter, interpret it as such
r = symbols('r')


w0 = V[0, 0] * A[1, 0] + V[0, 1] * A[1, 1] - V[1, 0] * A[0, 0] - V[1, 1] * A[0, 1]
w1 = V[0, 0] * A[0, 0] + V[0, 1] * A[0, 1] - V[1, 0] * A[1, 0] - V[1, 1] * A[1, 1]
w2 =-V[0, 0] * A[1, 0] - V[0, 1] * A[1, 1] - V[1, 0] * A[0, 0] - V[1, 1] * A[0, 1]
w3 = V[0, 0] * A[1, 0] - V[0, 1] * A[1, 1] - V[1, 0] * A[0, 0] + V[1, 1] * A[0, 1]
w4 = V[0, 0] * A[1, 1] + V[0, 1] * A[1, 0] - V[1, 0] * A[0, 1] - V[1, 1] * A[0, 0]
w5 = V[0, 0] * A[0, 0] - V[0, 1] * A[0, 1] - V[1, 0] * A[1, 0] + V[1, 1] * A[1, 1]
w6 = V[0, 0] * A[0, 1] + V[0, 1] * A[0, 0] - V[1, 0] * A[1, 1] - V[1, 1] * A[1, 0]
w7 = V[0, 0] * A[1, 0] - V[0, 1] * A[1, 1] + V[1, 0] * A[0, 0] - V[1, 1] * A[0, 1]
w8 = V[0, 0] * A[1, 1] + V[0, 1] * A[1, 0] + V[1, 0] * A[0, 1] + V[1, 1] * A[0, 0]

b1 = 1 / 16 * (1j * w5 + w6 + w7 - 1j * w8)
b2 = 1 / 16 * (1j * w5 - w6 + w7 + 1j * w8)
b3 = 1 / 8 * (1j * w1 - w2)

b4 = 1 / 8 * (-w3 + 1j * w4)
b5 = - 1 / 4 * w0

B = b1*r + b2 * conjugate(r) + b3
C = b4*r + conjugate(b4) * conjugate(r) + b5

H = 2j*conjugate(B)
Omega = C

theta = Symbol("θ", real=True)
field = im(H * exp(-2j * theta))
dynamics = Omega + field

#OA ODE
rho = symbols('ρ', real=True)
phi = symbols('ϕ', real=True)
R1 = im(b1 * exp(2j * phi) + b2)
R2 = im(b3 * exp(1j*phi))
I1 = re(b1 * exp(2j * phi) + b2)
I2 = re(b3 * exp(1j * phi))
I3 = re(b4 * exp(1j * phi))

rhodot = 2 * ( 1- rho**2) * (R1 * rho + R2)
phidot = 2 * ((rho**2 + 1)*I1 + (rho**2 + 1)/rho * I2 + 2*I3 * rho + re(b5))

#Example usecase for the last example in the current overleaf (12-16-2025)
a = Symbol("a", real=True)
b = Symbol("b", real=True)
c = Symbol("c", real=True)
d = Symbol("d", real=True)

e = Symbol("e", real=True)
f = Symbol("f", real=True)
g = Symbol("g", real=True)
h = Symbol("h", real=True)
#formatting this to be more indicative of the matrix structure
substitution_set = {V[0,0]: a, V[0,1]: b,
                    V[1,0]:c, V[1,1]:d,

                    A[0, 0]: e, A[0, 1]: f,
                    A[1, 0]: g, A[1, 1]: h}

#print(f"Real part of ib_1 is:{re(1j*b1.subs(substitution_set))}")
#print(f"abs(b2) is:{Abs(b2.subs(substitution_set))}")


#print(f"H(t) = {H.subs(substitution_set)}")
#print(f"Omega(t) = {Omega.subs(substitution_set)}")
#print(f"field(t) = {field.subs(substitution_set)}")
#print(f"dynamics(t) = {simplify(dynamics.subs(substitution_set))}")
print(f"rhodot(t) = {rhodot.subs(substitution_set)}")
print(f"phidot(t) = {phidot.subs(substitution_set)}")
