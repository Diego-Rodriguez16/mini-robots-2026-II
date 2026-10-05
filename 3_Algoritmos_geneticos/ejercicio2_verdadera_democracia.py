"""
Ejercicio 2 - Verdadera democracia
Reparto del poder (50 entidades con peso 1-100) entre 5 partidos de forma
proporcional a sus curules (50 curules repartidas al azar, distribucion uniforme).

Cromosoma : vector de 50 enteros en {0..4}; gen i = partido que recibe la entidad i.
Aptitud   : minimizar  sum_j | poder_j - objetivo_j |   (objetivo_j = poder_total * curules_j / 50)
Matriz de poder M (50 x 5): M[i, j] = peso_i si la entidad i va al partido j, 0 si no.

Ejecutar:  python ejercicio2_verdadera_democracia.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SEMILLA = 7
N_PARTIDOS, N_CURULES, N_ENTIDADES = 5, 50, 50
TAM_POB, N_GEN = 200, 400
P_CRUCE, P_MUT, TORNEO, ELITISMO = 0.9, 0.02, 3, 4

rng = np.random.default_rng(SEMILLA)
partidos = [f"Partido {c}" for c in "ABCDE"]

# ------------------ Datos del problema -----------------------------------
# Curules: composicion uniforme de 50 en 5 partes (cada partido >= 1 curul)
cortes = np.sort(rng.choice(np.arange(1, N_CURULES), size=N_PARTIDOS - 1, replace=False))
curules = np.diff(np.concatenate(([0], cortes, [N_CURULES])))
# Entidades: nombre y peso politico aleatorio entre 1 y 100
pesos = rng.integers(1, 101, size=N_ENTIDADES)
entidades = [f"Entidad {i + 1:02d}" for i in range(N_ENTIDADES)]

poder_total = pesos.sum()
objetivo = poder_total * curules / N_CURULES


# ------------------ AG ---------------------------------------------------
def poder_por_partido(pob):
    """(N, 50) -> (N, 5) poder acumulado por partido."""
    onehot = pob[:, :, None] == np.arange(N_PARTIDOS)[None, None, :]
    return (onehot * pesos[None, :, None]).sum(axis=1)


def error(pob):
    return np.abs(poder_por_partido(pob) - objetivo).sum(axis=1)


def torneo(err):
    n = len(err)
    cand = rng.integers(0, n, size=(n, TORNEO))
    return cand[np.arange(n), np.argmin(err[cand], axis=1)]


pob = rng.integers(0, N_PARTIDOS, size=(TAM_POB, N_ENTIDADES))
hist = []
for gen in range(N_GEN + 1):
    err = error(pob)
    hist.append({"generacion": gen, "mejor": err.min(), "promedio": err.mean()})
    if gen == N_GEN:
        break
    elite = pob[np.argsort(err)[:ELITISMO]].copy()
    padres = pob[torneo(err)]
    # cruce uniforme entre parejas consecutivas
    hijos = padres.copy()
    par = rng.random(TAM_POB // 2) < P_CRUCE
    mascara = rng.random((TAM_POB // 2, N_ENTIDADES)) < 0.5
    a, b = padres[0::2], padres[1::2]
    hijos[0::2] = np.where(par[:, None] & mascara, b, a)
    hijos[1::2] = np.where(par[:, None] & mascara, a, b)
    # mutacion: reasignar entidad a otro partido al azar
    mut = rng.random(hijos.shape) < P_MUT
    hijos = np.where(mut, rng.integers(0, N_PARTIDOS, size=hijos.shape), hijos)
    hijos[:ELITISMO] = elite
    pob = hijos

err = error(pob)
mejor = pob[np.argmin(err)]
hist = pd.DataFrame(hist)

# ------------------ Resultados -------------------------------------------
M = np.zeros((N_ENTIDADES, N_PARTIDOS), dtype=int)
M[np.arange(N_ENTIDADES), mejor] = pesos
matriz = pd.DataFrame(M, index=entidades, columns=partidos)
matriz.to_csv("/mnt/user-data/outputs/ejercicio2_matriz_poder.csv")

resumen = pd.DataFrame({
    "Curules": curules,
    "% curules": (100 * curules / N_CURULES).round(2),
    "Poder objetivo": objetivo.round(2),
    "Poder asignado": M.sum(axis=0),
    "% poder": (100 * M.sum(axis=0) / poder_total).round(2),
    "Diferencia": (M.sum(axis=0) - objetivo).round(2),
    "N° entidades": (M > 0).sum(axis=0),
}, index=partidos)
resumen.to_csv("/mnt/user-data/outputs/ejercicio2_resumen.csv")

print("=== Ejercicio 2 ===")
print(f"Poder total a repartir: {poder_total}")
print(resumen.to_string())
print(f"\nError absoluto total: {err.min():.2f}  "
      f"({100 * err.min() / (2 * poder_total):.3f}% del poder total mal repartido)")

# ------------------ Graficas ---------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(19, 7), gridspec_kw={"width_ratios": [1.1, 1.3, 0.9]})

x = np.arange(N_PARTIDOS)
axs[0].bar(x - 0.2, objetivo, 0.4, label="Objetivo (según curules)")
axs[0].bar(x + 0.2, M.sum(axis=0), 0.4, label="Asignado por el AG")
axs[0].set_xticks(x); axs[0].set_xticklabels([f"{p[-1]}\n({c} cur.)" for p, c in zip(partidos, curules)])
axs[0].set_title("Poder objetivo vs. asignado"); axs[0].set_ylabel("Puntos de poder")
axs[0].legend(); axs[0].grid(axis="y", alpha=0.3)

orden = np.argsort(mejor, kind="stable")
sns.heatmap(matriz.iloc[orden], ax=axs[1], cmap="YlOrRd", cbar_kws={"label": "Peso político"},
            yticklabels=True, linewidths=0.2, linecolor="white")
axs[1].set_title("Matriz de poder (50 entidades × 5 partidos)")
axs[1].tick_params(axis="y", labelsize=6)

axs[2].semilogy(hist.generacion, hist.mejor + 1e-3, label="Mejor")
axs[2].semilogy(hist.generacion, hist.promedio, label="Promedio")
axs[2].set_title("Convergencia (error absoluto, escala log)")
axs[2].set_xlabel("Generación"); axs[2].legend(); axs[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/ejercicio2_resultado.png", dpi=130)
print("Figura guardada: ejercicio2_resultado.png")
