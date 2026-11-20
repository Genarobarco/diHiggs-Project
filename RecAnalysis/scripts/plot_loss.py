import json
import matplotlib.pyplot as plt

OUT_TRAINING_PATH = '/home/usuario/VSCode/diHiggs-Project/OutTrainings/GN2Tracks_KGC_20261119-T212322'
N_EPOCHS = 50          # epochs que corrio el entrenamiento

TRAIN = f"{OUT_TRAINING_PATH}/train_loss VS step_chart_data.json"
VAL   = f"{OUT_TRAINING_PATH}/val_loss VS step_chart_data.json"

def leer(path):
    """Extrae (x, y) del export JSON de plotly: toma la traza con mas puntos."""
    with open(path) as f:
        fig = json.load(f)
    trazas = fig["data"] if isinstance(fig, dict) else fig
    t = max((d for d in trazas if d.get("x") and d.get("y")), key=lambda d: len(d["x"]))
    return t["x"], t["y"]

x_tr, y_tr = leer(TRAIN)
x_va, y_va = leer(VAL)

# step -> epoch: la curva de train abarca los N_EPOCHS completos
pasos_por_epoch = max(x_tr) / N_EPOCHS
ep_tr = [x / pasos_por_epoch for x in x_tr]
ep_va = [x / pasos_por_epoch for x in x_va]

print(f"{pasos_por_epoch:.1f} steps/epoch   "
      f"train: {len(ep_tr)} puntos hasta epoch {max(ep_tr):.1f}   "
      f"val: {len(ep_va)} puntos hasta epoch {max(ep_va):.1f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(ep_tr, y_tr, lw=1.5, label="train")
ax.plot(ep_va, y_va, lw=1.5, label="validation")

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUT_TRAINING_PATH}/loss_vs_epoch.png", dpi=200)
plt.show()