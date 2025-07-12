import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# TENTANDO REFAZER OS GRÁFICOS DO AMIT

# estados |0>, |1> e |2>, respectivamente.
ket_0 = basis(2,0)
ket_1 = fock(3,1)
ket_2 = fock(3,2)

# Vamos criar os operadores criação e aniquilação dependente do tempo
N = 1
J = 1.53e11  # Constante de acoplamento entre os dois guias.

a = tensor(destroy(N), qeye(N))  # operador do primeiro guia
b = tensor(qeye(N), destroy(N))  # operador do segundo guia


print(ket_0)