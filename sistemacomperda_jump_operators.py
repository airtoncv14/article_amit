import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Parâmetros
N = 3
J = 1.0
π = np.pi
hbar = 1.0
omega = 1

# Valores de theta para o gráfico
theta_vals = np.linspace(0, 3.1, 5000)
t_vals = (theta_vals * π) / J

# Operadores aniquilação
a = tensor(destroy(N), qeye(N))
b = tensor(qeye(N), destroy(N))

# (a†b + b†a): Parte da interação entre os dois guias.
G = a.dag() * b + b.dag() * a

# Hamiltoniano
H = omega * hbar * (a.dag() * a + b.dag() * b) + hbar * J * G

# Estado inicial
psi0 = tensor(basis(N,1), basis(N,1))

# Operador densidade
rho0 = ket2dm(psi0)

# Operador de salto em cada guia
def operadores_de_salto(alpha):
    gamma = 1
    gamma_G = gamma
    gamma_L = gamma * alpha
    c_ops = []
    c_ops.append(np.sqrt(gamma_G) * a.dag())  # ganho no guia 1
    c_ops.append(np.sqrt(gamma_L) * b)        # perda no guia 2
    return c_ops

# Evolução temporal com equação mestra de Lindblad
def rho_numerico(t, alpha):
    t_list = [t]
    result = mesolve(H, rho0, t_list, operadores_de_salto(alpha), [])
    return result.states[0]

# Calculando o log negativo
def log_negativo(rho):
    rho_pt = partial_transpose(rho, [0,1])
    evals = rho_pt.eigenenergies()
    negativity = sum(abs(e) for e in evals if e < 0)
    return np.log2(2 * negativity + 1)

# Parâmetros de alpha para o gráfico
alphas = [1.5, 1, 0.2]
cores = ['green', 'orange', 'red']
labels = [r'$\alpha = 1.5$', r'$\alpha = 1$', r'$\alpha = 0.2$']
log_neg_curvas = []

# Evolução para cada alpha
for alpha in alphas:
    result = mesolve(H, rho0, t_vals, operadores_de_salto(alpha), [])
    log_neg = [log_negativo(rho) for rho in result.states]
    log_neg_curvas.append(log_neg)


# Plotagem
plt.figure(figsize=(9, 5))
for i in range(len(alphas)):
    plt.plot(theta_vals, log_neg_curvas[i], label=labels[i], color=cores[i])

plt.xlabel(r'$\theta = \frac{Jt}{\pi}$', fontsize=14, fontname='Times New Roman')
plt.ylabel(r'$E_N->$', fontsize=14, fontname='Times New Roman')
plt.xlim(0, 0.8)
plt.ylim()
plt.title(r'Time evolution of the logarithmic negativity EN in presence of loss anf gain of the waveguide modes', fontsize=13)
plt.tick_params(which='both', direction='in', length=6, width=1)
plt.minorticks_on()
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()