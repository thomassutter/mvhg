import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from mvhg.pt_fmvhg import MVHG

if __name__ == "__main__":
    dir_results = os.path.join(".", "results")
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)
    class_names = ["violet", "yellow", "green"]
    colors = ["blueviolet", "yellow", "green"]
    m = torch.tensor([10, 10, 10])
    n = torch.tensor(10)
    log_w = torch.log(torch.tensor([1.0, 1.0, 1.0]))
    num_classes = 3
    num_samples = 1
    tau = 1.0
    fn_pre = "f"

    create_plot = True
    n = n.unsqueeze(0).repeat(num_samples, 1)
    log_w = log_w.unsqueeze(0).repeat(num_samples, 1)
    n_repeats = 1
    mvhg = MVHG(device="cpu", eps=1e-20)
    for h in range(n_repeats):
        y, x_all, y_m, log_p = mvhg(m, n, log_w, tau)
        log_p = mvhg.get_log_probs(m, n, x_all, y, log_w)

        str_ws = [str(w_j) for w_j in list(log_w[0].cpu().numpy().flatten())]
        str_weights = "_".join(str_ws)
        fn_plot = os.path.join(
            dir_results, "pt_samples_" + fn_pre + "_" + str_weights + ".png"
        )
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for j in range(num_classes):
            ind_j = np.arange(m[j].cpu().numpy() + 1)
            y_avg_j = y[j].mean(dim=0).cpu().numpy().flatten()
            ax.bar(ind_j, y_avg_j, alpha=0.5, color=colors[j], label=class_names[j])
        plt.title(str_ws)
        plt.legend()
        fig.tight_layout()
        plt.draw()
        plt.savefig(fn_plot, format="png")
