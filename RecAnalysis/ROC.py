import numpy as np
import pandas as pd
from ftag.hdf5 import H5Reader
from puma import Roc, RocPlot
from ftag.utils import calculate_rejection


# Functions
def disc_fct(arr: np.ndarray) -> np.ndarray:
    return arr[0]  # Esta es la forma del discriminante, pt, log(pt), lo que venga

def disc_rej(time_stamp, N_epoch):

    signal_name = f"/home/gbarco/DiHiggsProject/OutTrainings/GN2Tracks_v01_{time_stamp}/ckpts/epoch={N_epoch}-val_loss=0.00000__test_signal.h5"
    back_name = f"/home/gbarco/DiHiggsProject/OutTrainings/GN2Tracks_v01_{time_stamp}/ckpts/epoch={N_epoch}-val_loss=0.00000__test_background.h5"

    # Discriminante
    reader = H5Reader([signal_name, back_name], batch_size=1_000)

    df = pd.DataFrame(
        reader.load(
            {
                "jets": [
                    "pt",
                    "eta",
                    "flavour_label",
                    "GN2Tau_cells_phadtaujets",
                    "GN2Tau_cells_pqcdjets",
                ]
            }
        )["jets"]
    )

    discs_gn2 = np.apply_along_axis(
        disc_fct,
        1,
        df[
            [
                "GN2Tau_cells_phadtaujets",
                "GN2Tau_cells_pqcdjets",
            ]
        ].values,
    )

    # Procesamiento
    sig_eff = np.linspace(0.49, 1, 20)

    is_qcdjets = df["flavour_label"] == 1
    is_hadtaujets = df["flavour_label"] == 0

    n_jets_qcdjets = sum(is_qcdjets)
    n_jets_hadtaujets = sum(is_hadtaujets)

    gn2_taujets_rej = calculate_rejection(
        discs_gn2[is_hadtaujets],
        discs_gn2[is_qcdjets],
        sig_eff,
    )

    return gn2_taujets_rej


# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# Data

data_list = [
    ("20261104-T200117", "047"),
]

rejection_list = []

for tupla in data_list:

    time_stamp = tupla[0]
    N_epoch = tupla[1]

    rej_i = disc_rej(time_stamp, N_epoch)  # Calculo del determinante
    rejection_list.append(rej_i)


plot_roc = RocPlot(
    n_ratio_panels=1,
    ylabel="Background rejection",
    xlabel="$\\tau$-jet efficiency",
    atlas_second_tag="$\\sqrt{s}=13.6$ TeV \\\\ $\\gamma* \\to \\tau \\tau$ signal",
    figsize=(6.5, 6),
    y_scale=1.4,
)

for i, (time_stamp, N_epoch) in enumerate(data_list):

    sig_eff, rej, n_qcd = disc_rej(time_stamp, N_epoch)

    plot_roc.add_roc(
        Roc(
            sig_eff,
            rej,
            n_test=n_qcd,
            rej_class="qcd",
            signal_class="taujets",
            label=f"{time_stamp}",
        ),
        reference=(i == 0),
    )
