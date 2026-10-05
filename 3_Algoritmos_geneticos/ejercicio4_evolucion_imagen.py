"""
Ejercicio 4 - Evolucion de una imagen con un AG
Poblacion inicial: 50 matrices 120 x 180 x 3 (RGB) con enteros aleatorios 0..255.
Funcion de aptitud: parecido a una imagen objetivo -> se minimiza el error cuadratico medio (MSE).

Operadores:
  * Seleccion : torneo (k=3) sobre el MSE de la imagen completa.
  * Cruce     : uniforme PIXEL A PIXEL, guiado por la aptitud local. Como el MSE es la suma de
                los errores de cada pixel, en cada pixel el hijo hereda (con prob. 0.9) el valor
                del padre que esta mas cerca del objetivo; con prob. 0.1 elige al azar.
  * Mutacion  : gaussiana dispersa (solo una fraccion de los genes), con sigma y tasa que
                disminuyen a lo largo de la evolucion (de exploracion a refinamiento).
  * Elitismo  : los 2 mejores pasan intactos.

Ejecutar:
    python ejercicio4_evolucion_imagen.py                   # imagen de ejemplo (astronauta)
    python ejercicio4_evolucion_imagen.py mi_foto.png       # tu propia imagen
    python ejercicio4_evolucion_imagen.py mi_foto.png 1500  # ...y numero de generaciones
    python ejercicio4_evolucion_imagen.py - 1500            # imagen de ejemplo + generaciones
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from skimage import data
from skimage.transform import resize
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim

# ----------------------------- Parametros ---------------------------------
SEMILLA = 1
ALTO, ANCHO, CANALES = 120, 180, 3
TAM_POB = 50
N_GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
TORNEO = 3
ELITISMO = 2
P_GUIADO = 0.9                       # prob. de que el cruce use la aptitud local
SIGMA_INI, SIGMA_FIN = 30.0, 1.0     # desviacion de la mutacion gaussiana
PMUT_INI, PMUT_FIN = 0.05, 0.01      # fraccion de genes mutados por generacion

rng = np.random.default_rng(SEMILLA)

# ----------------------------- Imagen objetivo ----------------------------
if len(sys.argv) > 1 and sys.argv[1] != "-":
    img = imread(sys.argv[1])[..., :3]
else:
    img = data.astronaut()
objetivo = (resize(img, (ALTO, ANCHO), anti_aliasing=True) * 255).round().astype(np.float32)


def psnr(m):
    return 10 * np.log10(255.0 ** 2 / max(m, 1e-9))


# ----------------------------- Poblacion inicial --------------------------
pob = rng.integers(0, 256, size=(TAM_POB, ALTO, ANCHO, CANALES)).astype(np.float32)
poblacion_inicial = pob.astype(np.uint8)

instantes = sorted({0, 2, 5, 10, 25, 50, 100, 300, N_GEN})
instantes = [g for g in instantes if g <= N_GEN]
capturas, hist_mejor, hist_prom = {}, [], []
idx = np.arange(TAM_POB)

for gen in range(N_GEN + 1):
    sq = (pob - objetivo) ** 2                 # (N, H, W, 3) error cuadratico por gen
    err = sq.mean(axis=(1, 2, 3))              # aptitud (MSE) de cada individuo
    i_mejor = int(np.argmin(err))
    hist_mejor.append(err[i_mejor]); hist_prom.append(err.mean())
    if gen in instantes:
        capturas[gen] = pob[i_mejor].astype(np.uint8)
    if gen % 100 == 0 or gen == N_GEN:
        print(f"gen {gen:5d} | MSE mejor = {err[i_mejor]:9.2f} | PSNR = {psnr(err[i_mejor]):5.2f} dB")
    if gen == N_GEN:
        break

    t = gen / N_GEN
    sigma = SIGMA_INI * (SIGMA_FIN / SIGMA_INI) ** t
    p_mut = PMUT_INI * (PMUT_FIN / PMUT_INI) ** t

    # Seleccion por torneo de dos padres por hijo
    cand = rng.integers(0, TAM_POB, size=(2, TAM_POB, TORNEO))
    pa = cand[0][idx, np.argmin(err[cand[0]], axis=1)]
    pb = cand[1][idx, np.argmin(err[cand[1]], axis=1)]

    # Cruce uniforme pixel a pixel guiado por el error local
    local = sq.sum(axis=3)                                     # (N, H, W)
    a_es_mejor = local[pa] <= local[pb]
    guiado = rng.random((TAM_POB, ALTO, ANCHO)) < P_GUIADO
    tomar_a = np.where(guiado, a_es_mejor, rng.random((TAM_POB, ALTO, ANCHO)) < 0.5)
    hijos = np.where(tomar_a[..., None], pob[pa], pob[pb])

    # Mutacion gaussiana dispersa (ciega: no mira el objetivo)
    n_mut = rng.binomial(hijos.size, p_mut)
    pos = rng.integers(0, hijos.size, n_mut)
    hijos.reshape(-1)[pos] += rng.normal(0, sigma, n_mut).astype(np.float32)
    hijos = np.rint(np.clip(hijos, 0, 255))

    # Elitismo
    hijos[:ELITISMO] = pob[np.argsort(err)[:ELITISMO]]
    pob = hijos

mejor = capturas[N_GEN]
similitud = ssim(mejor, objetivo.astype(np.uint8), channel_axis=2)
print(f"\nResultado final: MSE = {hist_mejor[-1]:.3f} | PSNR = {psnr(hist_mejor[-1]):.2f} dB "
      f"| SSIM = {similitud:.4f}")

# ----------------------------- Graficas -----------------------------------
# 1) Poblacion inicial como graficas RGB
fig, axs = plt.subplots(2, 5, figsize=(14, 6))
for k, ax in enumerate(axs.ravel()):
    ax.imshow(poblacion_inicial[k]); ax.set_title(f"Individuo {k + 1}", fontsize=9); ax.axis("off")
fig.suptitle("Población inicial: 10 de las 50 matrices aleatorias (120×180×3, valores 0–255)")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/ejercicio4_poblacion_inicial.png", dpi=110)
plt.close(fig)

# 2) Evolucion del mejor individuo
n = len(instantes)
cols = (n + 2) // 2
fig, axs = plt.subplots(2, cols, figsize=(3.2 * cols, 5.6))
axs = axs.ravel()
for k, g in enumerate(instantes):
    axs[k].imshow(capturas[g]); axs[k].set_title(f"Gen {g}\nMSE = {hist_mejor[g]:.1f}", fontsize=9)
axs[n].imshow(objetivo.astype(np.uint8)); axs[n].set_title("OBJETIVO", fontsize=9, color="tab:green")
for ax in axs:
    ax.axis("off")
fig.suptitle("Evolución del mejor individuo hacia la imagen objetivo")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/ejercicio4_evolucion.png", dpi=110)
plt.close(fig)

# 3) Convergencia
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.semilogy(np.array(hist_mejor) + 1e-3, label="Mejor MSE")
ax.semilogy(np.array(hist_prom) + 1e-3, label="MSE promedio", alpha=0.7)
ax.set_xlabel("Generación"); ax.set_ylabel("MSE (escala log)")
ax.set_title("Convergencia del AG"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/ejercicio4_convergencia.png", dpi=120)
plt.close(fig)

# 4) GIF de la evolucion
cuadros = [np.kron(capturas[g], np.ones((3, 3, 1), dtype=np.uint8)) for g in instantes]
imageio.mimsave("/mnt/user-data/outputs/ejercicio4_evolucion.gif", cuadros, duration=0.9, loop=0)
print("Archivos guardados: ejercicio4_poblacion_inicial.png, _evolucion.png, _convergencia.png, _evolucion.gif")
