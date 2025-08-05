import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Número de guias
N = 10

# Dimensão de cada guia (0 ou 1 fóton)
dim = 2

# Frequência de propagação
omega = 1

# Lista de tempo
tlist = np.linspace(0, 10, 1000)

# Coeficientes de acoplamento entre guias vizinhos. Aqui temos o acoplamento fixo e aleatório (fica a seu critério)
#J = np.random.uniform(low=0.0, high=2.0, size=N - 1)
J = [1.0] * (N - 1)

# Operador identidade
I = qeye(dim)

# Função que cria o operador de aniquilação no guia k
def ak(k):
    ops = [I] * N
    ops[k] = destroy(dim)
    return tensor(ops)


# ---------- ESCREVENDO O HAMILTONIANO ----------

H = 0
for k in range(N):
    a_k = ak(k)

    # Termo local: ω a† a
    H += omega * a_k.dag() * a_k

for k in range(N - 1):
    a_k = ak(k)
    a_k1 = ak(k + 1)

    # Termo de interação entre os guias
    H += J[k] * (a_k.dag() * a_k1 + a_k * a_k1.dag())

# ---------- FIM ----------


# ---------- ESCREVENDO O ESTADO INICIAL ----------

# Índices que representa os guias em que serão colocados os fótons
m = 0
n = 1

# Estado de base com 0 e 1 fóton.
ket_0 = basis(dim, 0)
ket_1 = basis(dim, 1)

# Lista para armazenar os dados do estado inicial
estados = []

# Função para que seja possível escrever o estado inicial, onde 2 guias terão o estado |1> e o resto zero.
for k in range(N):
    if k == m or k == n:
        estados.append(ket_1)
    else:
        estados.append(ket_0)

# Estado inicial
psi0 = tensor(estados)

# ---------- FIM ----------


# Evolução temporal 
result = mesolve(H, psi0, tlist, [], [])

# Operador densisade ρ(t)=∣ψ(t)⟩⟨ψ(t)∣
psi_t_list = result.states
rho_t_list = [ket2dm(psi) for psi in psi_t_list]


# ---------- CÁLCULO DO EMARANHAMENTO PARA TODOS OS PARES DE GUIAS (p, q) AO LONGO DO TEMPO ----------

# Inicializar matriz 3D (t, p, q). Pois, para cada tempo, queremos saber quais pares de guias estão emaranhados e com que intensidade
emaranhamento = np.zeros((len(tlist), N, N))

for t_idx, rho_t in enumerate(rho_t_list):
    for p in range(N):
        for q in range(p+1, N):  # só pares p < q para evitar repetição

            # Traço do operador densidade total sobre todo subsistema menos para os guias p e q
            rho_pq = rho_t.ptrace([p, q])

            # Transposta parcial no subsistema p
            rho_pq_pt = partial_transpose(rho_pq, [0, 1])

            # Norma traço
            trace_norm = rho_pq_pt.norm('tr')

            # Logaritmo negativo 
            log_neg = np.log2(trace_norm)

            # Armazena o valor do logaritmo negativo entre os guias p e q 
            emaranhamento[t_idx, p, q] = log_neg

            # para garantir que seja simétrico
            emaranhamento[t_idx, q, p] = log_neg  
# ---------- FIM ----------


# ---------- PLOTAGEM DO COLORMAP EM UM TEMPO ESPECÍFICO T----------

# Defindo um tempo desejado
t = 3.3
# Índice do tempo mais próximo
t_index = np.argmin(np.abs(tlist - t))  

plt.figure(figsize=(6,5))
plt.imshow(emaranhamento[t_index], origin='lower', cmap='viridis')
plt.colorbar(label='Emaranhamento $E_N$')
plt.title(f'Emaranhamento entre guias no tempo t ≈ {tlist[t_index]:.2f}')
plt.xlabel('q')
plt.ylabel('p')
plt.tight_layout()
plt.show()
# ---------- FIM ----------