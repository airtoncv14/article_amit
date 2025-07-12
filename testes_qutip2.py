import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Parâmetros
π = np.pi
hbar = 1.0
omega = 1.0
J = 1.53e11

# Tempo e theta
theta_vals = np.linspace(0, 3.1, 5000)
t_vals = (theta_vals * π) / J

# Espaço de Hilbert e operadores
N = 3

# Operadores de aniquilação
a = tensor(destroy(N), qeye(N))  # operador 'a' no primeiro guia
b = tensor(qeye(N), destroy(N))  # operador 'b' no segundo guia

# Hamiltoniano
H = hbar * omega * (a.dag() * a + b.dag() * b) + J * (a.dag() * b + b.dag() * a)

# Função para calcular emaranhamento logarítmico
def calcular_log_neg(estado_inicial):
    psi_t = mesolve(H, estado_inicial, t_vals, [], []).states
    return [np.log2(partial_transpose(ket2dm(psi), [0, 1], 1).norm('tr')) for psi in psi_t]

# Estados iniciais
psi_11 = tensor(basis(N, 1), basis(N, 1))
psi_20 = tensor(basis(N, 2), basis(N, 0))

# Emaranhamentos
EN_11 = calcular_log_neg(psi_11)
EN_20 = calcular_log_neg(psi_20)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(theta_vals, EN_11, color='red', label=r'$E_N(t)$')
plt.plot(theta_vals, EN_20, color='black', linewidth=2.5, label=r'$E_2(\theta)$ - Dois fótons no guia 1')
plt.xlabel(r'$\theta = \frac{Jt}{\pi}$', fontsize=14, fontname='Times New Roman')
plt.ylabel(r'$E_N$', fontsize=14, fontname='Times New Roman')
plt.xlim(0, 3.2)
plt.ylim(0, 1.7)
plt.title('Time evolution of log negativity for separable input states')
plt.tick_params(which='both', direction='in', length=6, width=1)
plt.minorticks_on()
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()
