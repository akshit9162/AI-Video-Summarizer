import matplotlib.pyplot as plt
import numpy as np

def plot_highlights(imp, idxs):
    x = np.arange(len(imp))
    plt.figure()
    plt.plot(x, imp)
    plt.scatter(idxs, imp[idxs])
    plt.title("Highlights")
    plt.xlabel("Frame")
    plt.ylabel("Importance")
    plt.savefig("highlights.png")
    return "highlights.png"