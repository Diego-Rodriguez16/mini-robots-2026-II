"""
Ejercicio 1 - Maximizar f(x) = x*sen(10*pi*x) + 1,  x en [0, 1]
Algoritmo genetico con codificacion binaria (NumPy + Matplotlib + Pandas).

Ejecutar:  python ejercicio1_maximizar_funcion.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- Parametros ---------------------------------
SEMILLA = 42
N_BITS = 22            # resolucion = 1 / (2^22 - 1) ~ 2.4e-7
TAM_POB = 100
N_GEN = 200
P_CRUCE = 0.9
P_MUT = 2.0 / N_BITS   # tasa de mutacion por bit
TORNEO = 3
ELITISMO = 2

rng = np.random.default_rng(SEMILLA)


# ----------------------------- Funcion objetivo ---------------------------
def f(x):
    return x * np.sin(10 * np.pi * x) + 1


def decodificar(pob):
    """Matriz de bits (N, N_BITS) -> reales en [0, 1]."""
    pesos = 2 ** np.arange(N_BITS - 1, -1, -1)
    enteros = pob @ pesos
    return enteros / (2 ** N_BITS - 1)


# ----------------------------- Operadores geneticos -----------------------
def seleccion_torneo(aptitud):
    n = len(aptitud)
    candidatos = rng.integers(0, n, size=(n, TORNEO))
    ganadores = candidatos[np.arange(n), np.argmax(aptitud[candidatos], axis=1)]
    return ganadores


def cruce_un_punto(padres):
    n, L = padres.shape
    hijos = padres.copy()
    for i in range(0, n - 1, 2):
        if rng.random() < P_CRUCE:
            punto = rng.integers(1, L)
            hijos[i, punto:], hijos[i + 1, punto:] = (
                padres[i + 1, punto:].copy(),
                padres[i, punto:].copy(),
            )
    return hijos


def mutacion(pob):
    mascara = rng.random(pob.shape) < P_MUT
    return np.where(mascara, 1 - pob, pob)


# ----------------------------- Bucle evolutivo ----------------------------
pob = rng.integers(0, 2, size=(TAM_POB, N_BITS))
historial = []
instantaneas = {}

for gen in range(N_GEN + 1):
    x = decodificar(pob)
    apt = f(x)
    i_mejor = np.argmax(apt)
    historial.append(
        {"generacion": gen, "mejor": apt[i_mejor], "promedio": apt.mean(),
         "peor": apt.min(), "x_mejor": x[i_mejor]}
    )
    if gen in (0, 5, 20, N_GEN):
        instantaneas[gen] = x.copy()
    if gen == N_GEN:
        break

    elite = pob[np.argsort(apt)[-ELITISMO:]].copy()
    padres = pob[seleccion_torneo(apt)]
    hijos = mutacion(cruce_un_punto(padres))
    hijos[:ELITISMO] = elite
    pob = hijos

df = pd.DataFrame(historial)
df.to_csv("/mnt/user-data/outputs/ejercicio1_historial.csv", index=False)

# ----------------------------- Verificacion -------------------------------
xs = np.linspace(0, 1, 2_000_001)
x_real = xs[np.argmax(f(xs))]
f_real = f(x_real)
mejor = df.iloc[-1]

print("=== Ejercicio 1 ===")
print(f"AG        : x* = {mejor.x_mejor:.6f}   f(x*) = {mejor.mejor:.6f}")
print(f"Busqueda  : x* = {x_real:.6f}   f(x*) = {f_real:.6f}  (fuerza bruta, 2M puntos)")
print(f"Error en f: {abs(f_real - mejor.mejor):.2e}")

# ----------------------------- Graficas -----------------------------------
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

xx = np.linspace(0, 1, 2000)
axs[0].plot(xx, f(xx), color="0.3", lw=1.2, label="f(x)")
colores = {0: "tab:red", 5: "tab:orange", 20: "tab:green", N_GEN: "tab:blue"}
for g, xg in instantaneas.items():
    axs[0].scatter(xg, f(xg), s=22, color=colores[g], alpha=0.75,
                   label=f"Poblacion gen {g}", zorder=3)
axs[0].scatter([mejor.x_mejor], [mejor.mejor], marker="*", s=300, color="gold",
               edgecolor="k", zorder=5, label=f"Mejor: x={mejor.x_mejor:.4f}")
axs[0].set_title("f(x) = x sen(10πx) + 1  y evolución de la población")
axs[0].set_xlabel("x"); axs[0].set_ylabel("f(x)"); axs[0].legend(fontsize=8)
axs[0].grid(alpha=0.3)

axs[1].plot(df.generacion, df.mejor, label="Mejor", lw=2)
axs[1].plot(df.generacion, df.promedio, label="Promedio", lw=1.5)
axs[1].plot(df.generacion, df.peor, label="Peor", lw=1, alpha=0.6)
axs[1].axhline(f_real, ls="--", color="k", lw=0.8, label="Óptimo real")
axs[1].set_title("Convergencia del AG")
axs[1].set_xlabel("Generación"); axs[1].set_ylabel("Aptitud")
axs[1].legend(); axs[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/ejercicio1_resultado.png", dpi=130)
print("Figura guardada: ejercicio1_resultado.png")
