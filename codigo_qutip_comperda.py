import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Parâmetros
N = 3
J = 1.0
π = np.pi

# Valores de theta para o gráfico
theta_vals = np.linspace(0, 0.76, 5000)
t_vals = (theta_vals * π) / J

# Operadores aniquilação
a = tensor(destroy(N), qeye(N))
b = tensor(qeye(N), destroy(N))

# (a†b + b†a): Parte da interação entre os dois guias.
G = a.dag() * b + b.dag() * a

# Montando a solução para a matriz densidade da equação (28) do artigo.
def rho_til_analitico(t, gamma):
    exponencial = np.exp(2 * gamma * t)
    coef_00 = (exponencial - 1)**2 * np.exp(-4 * gamma * t)  # coeficiente que multiplica |0,0><0,0|
    coef_10_01 = (exponencial - 1) * np.exp(-4 * gamma * t)  # coeficiente que multiplica (|1,0><1,0| + |0,1><0,1|)
    coef_11 = np.exp(-4 * gamma * t)                         # coeficiente que multiplica |1,1><1,1|

    ket_00 = tensor(basis(N, 0), basis(N, 0))  # |0,0>
    ket_10 = tensor(basis(N, 1), basis(N, 0))  # |1,0>
    ket_01 = tensor(basis(N, 0), basis(N, 1))  # |0,1>
    ket_11 = tensor(basis(N, 1), basis(N, 1))  # |1,1>

    proj_00 = ket_00 * ket_00.dag()  # |0,0><0,0|
    proj_10 = ket_10 * ket_10.dag()  # |1,0><1,0|
    proj_01 = ket_01 * ket_01.dag()  # |0,1><0,1|
    proj_11 = ket_11 * ket_11.dag()  # |1,1><1,1|

    # Solução ρ˜(t)
    rho_til = coef_00 * proj_00 + coef_10_01 * (proj_10 + proj_01) + coef_11 * proj_11
    return rho_til


# Escrevendo ρ(t) em termos de ρ˜(t)
def rho_final(t, gamma):  # Essa função aplica a rotação unitária em ρ˜(t) para obter ρ(t)

    # Esta chamando a solução de ρ˜(t) obtida na função anterior "rho_til_analitico(t, gamma)" para construir a matriz densidade ρ(t) da equação (29) do artigo
    rho_til = rho_til_analitico(t, gamma)

    # U(t)
    U = (-1j * J * t * G).expm()

    # U†(t)
    U_dag = U.dag()

    return U * rho_til * U_dag  #  ρ(t)=Uρ˜(t)U†


# Função para calcular o logarítimo negativo em ρ(t)
def negatividade_log(rho):

    # Transposta parcial no segundo modo.
    rho_pt = partial_transpose(rho, [0, 1])

    # Cálculo dos autovalores da transposta parcial de ρ(t)
    evals = rho_pt.eigenenergies()

    # Cálculo do logarítimo negativo da transposta parcial de ρ(t)
    negativity = sum(abs(e) for e in evals if e < 0)
    return np.log2(2 * negativity + 1)

# Calcular para diferentes gammas (só peguei e colei aqui).
gammas = [0.1 * J, 0.2 * J, 0.3 * J]
cores = ['green', 'orange', 'red']
labels = [r'$\gamma/J = 0.1$', r'$\gamma/J = 0.2$', r'$\gamma/J = 0.3$']
log_neg_curvas = []

for gamma in gammas:
    log_neg = [negatividade_log(rho_final(t, gamma)) for t in t_vals]
    log_neg_curvas.append(log_neg)

# Plotagem
plt.figure(figsize=(9, 5))
for i in range(3):
    plt.plot(theta_vals, log_neg_curvas[i], label=labels[i], color=cores[i])

plt.xlabel(r'$\theta = \frac{Jt}{\pi}$', fontsize=14, fontname='Times New Roman')
plt.ylabel(r'$E_N->$', fontsize=14, fontname='Times New Roman')
plt.xlim(0, 0.77)
plt.ylim(0, 1.4)
plt.title(r'Time evolution of the logarithmic negativity EN in presence of loss of the waveguide modes', fontsize=13)
plt.tick_params(which='both', direction='in', length=6, width=1)
plt.minorticks_on()
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()