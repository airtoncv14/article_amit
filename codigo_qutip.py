import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Parâmetros do artigo
π = np.pi
hbar = 1.0
omega = 1.0
J = 1

# Geração dos valores de tempo e de θ para o gráfico
theta_vals = np.linspace(0, 3.2, 5000)
t_vals = (theta_vals * π) / J

# ============
# PARTE 1 - ESTADOS |1,1> E |2,0>
# ============

N = 12  # dimensão suficientemente grande
a = tensor(destroy(N), qeye(N))  # operador 'a' no primeiro guia
b = tensor(qeye(N), destroy(N))  # operador 'b' no segundo guia

# Hamiltoniano
H = hbar * omega * (a.dag() * a + b.dag() * b) + J * (a.dag() * b + b.dag() * a)

# Estados iniciais
psi_11 = tensor(basis(N, 1), basis(N, 1))  # Estado inicial |1,1>
psi_20 = tensor(basis(N, 2), basis(N, 0))  # Estado inicial |2,0>

# Função para calcular negatividade logarítmica
def calcular_log_neg(estado_inicial, H, t_vals):

    # Evolução temporal
    psi_t = mesolve(H, estado_inicial, t_vals, [], []).states

    return [np.log2(partial_transpose(ket2dm(psi), [0, 1], 1).norm('tr')) for psi in psi_t]

# Vai pegar essas variáveis e jogar na função acima para calcular o emaranhamento.
EN_11 = calcular_log_neg(psi_11, H, t_vals)
EN_20 = calcular_log_neg(psi_20, H, t_vals)

# Gráfico 1: estados de fock
plt.figure(figsize=(8, 5))
plt.plot(theta_vals, EN_11, color='red', label=r'$|1,1\rangle$')
plt.plot(theta_vals, EN_20, color='black', linewidth=2.5, label=r'$|2,0\rangle$')
plt.xlabel(r'$\theta = \frac{Jt}{\pi}$', fontsize=14, fontname='Times New Roman')
plt.ylabel(r'$E_N->$', fontsize=14, fontname='Times New Roman')
plt.xlim(0, 3.2)
plt.ylim(0, 1.7)
plt.title('Time evolution of log negatively for the separable input state')
plt.tick_params(which='both', direction='in', length=6, width=1)
plt.minorticks_on()
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()

# ============
# PARTE 2 - ESTADOS NOON: para 2 fótons e 4 fótons
# ============

# Função para criar operadores e Hamiltoniano para uma dada dimensão
def criar_hamiltoniano(dim):
    a = tensor(destroy(dim), qeye(dim))
    b = tensor(qeye(dim), destroy(dim))
    H = hbar * omega * (a.dag() * a + b.dag() * b) + J * (a.dag() * b + b.dag() * a)
    return H

# Função para criar o estado NOON e Hamiltoniano para n fótons
def criar_estado_NOON(n):
    dim = n + 1
    ket_N0 = tensor(basis(dim, n), basis(dim, 0))  # |N,0>
    ket_0N = tensor(basis(dim, 0), basis(dim, n))  # |0,N>
    psi_NOON = (ket_N0 + ket_0N).unit()
    H = criar_hamiltoniano(dim)
    return psi_NOON, H

# Logarítimo negativo para o estado NOON de n fótons
def negatividade_NOON(n):
    psi_NOON, H = criar_estado_NOON(n)
    t_vals = (theta_vals * π) / J
    return calcular_log_neg(psi_NOON, H, t_vals)

# Vai pegar essas variáveis e jogar na função acima para calcular o emaranhamento.
EN_NOON_2 = negatividade_NOON(2)               # Curva preta: E_N para N=2
EN_NOON_4 = negatividade_NOON(4)               # Curva vermelha: (E_N - 1) para N=4

EN_NOON_4_menos1 = [x - 1 for x in EN_NOON_4]  # Subtrai 1 de cada ponto

# Gráfico 2: estados NOON
plt.figure(figsize=(8, 5))
plt.plot(theta_vals, EN_NOON_2, color='black', linewidth=2, label=r'$NOON$ -> 2 photons')
plt.plot(theta_vals, EN_NOON_4_menos1, color='red', linewidth=2, label=r'$NOON$ -> 4 photons')
plt.xlabel(r'$\theta = \frac{Jt}{\pi}$', fontsize=14, fontname='Times New Roman')
plt.ylabel(r'$E_N->$', fontsize=14, fontname='Times New Roman')
plt.xlim(0, 3.2)
plt.ylim(0, 1.7)
plt.title('The behavior of log negatively for the NOON state')
plt.tick_params(which='both', direction='in', length=6, width=1)
plt.minorticks_on()
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()