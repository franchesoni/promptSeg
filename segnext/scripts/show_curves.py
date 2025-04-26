import pickle

import matplotlib.pyplot as plt
import numpy as np

curves = [
    ("sam multiscale stab thresh", "all_curves_hypersim_sam.pickle"),
    ("sam multiscale", "all_curves_hypersim_sam_nostabthresh.pickle"),
    ("sam 1mask", "all_curves_hypersim_sam1mask.pickle"),
    ("segnext ft", "all_curves_hypersim_segnextft.pickle"),
    ("segnext coco", "all_curves_hypersim_segnext.pickle"),

]
# Prepare both curves
def compute_mean_std(curves):
    max_len = max(len(curve) for curve in curves)
    curves_padded = [np.pad(curve, (0, max_len - len(curve)), constant_values=curve[-1]) for curve in curves]
    curves_arr = np.stack(curves_padded)
    mean_curve = np.nanmean(curves_arr, axis=0)
    std_curve = np.nanstd(curves_arr, axis=0)
    return mean_curve, std_curve, curves_arr

plt.figure()
for name, curve_path in curves:
    with open(curve_path, 'rb') as f:
        all_curves = pickle.load(f)
    mean_curve, std_curve, _ = compute_mean_std(all_curves)
    x = np.arange(1, len(mean_curve)+1)
    plt.plot(x, mean_curve, label=name)
    plt.fill_between(x, mean_curve-std_curve, mean_curve+std_curve, alpha=0.3)

plt.xlabel("Number of masks")
plt.ylabel("Mean IoU (model order)")
plt.title("Zero-shot mIoU vs. number of masks (model order)")
plt.grid()
plt.legend()
plt.savefig("mean_miou_curve.png")
