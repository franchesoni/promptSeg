import pickle

import matplotlib.pyplot as plt
import numpy as np

all_curves_sam = pickle.load(open('all_curves_hypersim_sam.pickle', 'rb'))
all_curves_segnext = pickle.load(open('all_curves_hypersim_segnextft.pickle', 'rb'))

# Prepare both curves
def compute_mean_std(curves):
    max_len = max(len(curve) for curve in curves)
    curves_padded = [np.pad(curve, (0, max_len - len(curve)), constant_values=curve[-1]) for curve in curves]
    curves_arr = np.stack(curves_padded)
    mean_curve = np.nanmean(curves_arr, axis=0)
    std_curve = np.nanstd(curves_arr, axis=0)
    return mean_curve, std_curve, curves_arr

mean_curve_sam, std_curve_sam, curves_arr_sam = compute_mean_std(all_curves_sam)
mean_curve_segnext, std_curve_segnext, curves_arr_segnext = compute_mean_std(all_curves_segnext)

print("Mean mIoU curve (SAM):", mean_curve_sam)
print("Mean mIoU curve (SegNext):", mean_curve_segnext)

plt.figure()
x_sam = np.arange(1, len(mean_curve_sam)+1)
x_segnext = np.arange(1, len(mean_curve_segnext)+1)
plt.plot(x_sam, mean_curve_sam, label="SAM Mean mIoU")
plt.fill_between(x_sam, mean_curve_sam-std_curve_sam, mean_curve_sam+std_curve_sam, alpha=0.3, label="SAM Std Dev")
plt.plot(x_segnext, mean_curve_segnext, label="SegNext Mean mIoU")
plt.fill_between(x_segnext, mean_curve_segnext-std_curve_segnext, mean_curve_segnext+std_curve_segnext, alpha=0.3, label="SegNext Std Dev")
plt.xlabel("Number of masks")
plt.ylabel("Mean IoU (model order)")
plt.title("Zero-shot mIoU vs. number of masks (model order)")
plt.grid()
plt.legend()
plt.savefig("mean_miou_curve.png")
