import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Parâmetros 
π = np.pi 
hbar = 1.0          # Usamos hbar = 1 para simplificar
omega = 1.0         # Frequência dos modos
J = 1.53e11         # Constante de acoplamento entre os dois guias.

# Vamos encontrar o intervalor de tempo(t) de forma que θ = Jt/π,  vá de 0 à 3.1. Pois, no artigo, o gráfico vai de 0 à 3.1 no eixo X.
theta_max = 3.1
t_max = (theta_max * π ) / J
t_valores = np.linspace(0, t_max, 5000)   # Vai de 0 à t_max, onde terão 5000 valores.
theta_vals = (J * t_valores) / π  # de t para theta

# REALIZANDO O CÁLCULO DO EMARANHAMENTO.
N = 3  

# Operadores de aniquilação
a = tensor(destroy(N), qeye(N))  # operador 'a' no primeiro guia
b = tensor(qeye(N), destroy(N))  # operador 'b' no segundo guia

# Hamiltoniano: H = ω(a†a + b†b) + J(a†b + b†a)
H = hbar * omega * (a.dag() * a + b.dag() * b) + J * (a.dag() * b + b.dag() * a)

# Estado inicial |1,1>
psi_0 = tensor(basis(N, 1), basis(N, 1))      

# Evolução temporal
psi_t = mesolve(H, psi_0, t_valores, [], [])  

# Lista de operadores densidade ρ(t)
rho_list = [ket2dm(psi) for psi in psi_t.states]   

# Cálculo do emaranhamento - Logarítimo negativo via norma traço da transposta parcial
EN_list = [] # dicionário que armazena os resultados obtidos.

for rho in rho_list: 

    rho_PT = partial_transpose(rho, [0, 1], 1)   # Transposta parcial no segundo subsistema (guia 2)
    norm_trace = rho_PT.norm('tr')  # Norma traço da matriz da transposta parcial
    EN = np.log2(norm_trace)   # Emaranhamento
    EN_list.append(EN)   # Armazendo o resultado na lista


#PARA O ESTADO |2,0>
# Estado inicial |2,0>
psi2_0 = tensor(basis(N, 2), basis(N, 0))   

# Evolução temporal
psi2_t = mesolve(H, psi2_0, t_valores, [], [])   

# Lista de operadores densidade ρ(t)
rho_list_2 = [ket2dm(psi2) for psi2 in psi2_t.states]   

# Cálculo do emaranhamento - Logarítimo negativo via norma traço da transposta parcial
EN_list2 = [] # dicionário que armazena os resultados obtidos.

for rho2 in rho_list_2: 

    rho_PT_2 = partial_transpose(rho2, [0, 1], 1)   # Transposta parcial no segundo subsistema (guia 2)
    norm_trace2 = rho_PT_2.norm('tr')  # Norma traço da matriz da transposta parcial
    EN_2 = np.log2(norm_trace2)   # Emaranhamento
    EN_list2.append(EN_2)   # Armazendo o resultado na lista

# Plotar EN(t) vs Θ
plt.figure(figsize=(8, 5))
plt.plot(theta_vals, EN_list, color='red', label=r'$E_N(t)$')
plt.plot(theta_vals, EN_list2, color='black', linewidth=2.5, label='$E_2(\\theta)$-Dois fótons no guia 1') 
plt.xlabel(r'$\theta = \frac{Jt}{\pi}$', fontsize=14, fontname='Times New Roman')
plt.ylabel(r'$E_N$', fontsize=14, fontname='Times New Roman')
plt.xlim(0, 3.2)
plt.ylim(0, 1.7)
plt.title('Time evolution of log negatively for the separable input state')
plt.tick_params(which='both', direction='in', length=6, width=1)
plt.minorticks_on()
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()