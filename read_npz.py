import numpy as np
import matplotlib.pyplot as plt

# 1) Veriyi yükle
data = np.load("Features.npz")
Features = data["Features"]

print("Orijinal boyut:", Features.shape)

# 2) İlk 100 kişiyi al
Features = Features[:, :100, :]
print("Yeni boyut:", Features.shape)

# 3) Normalize et
min_val = Features.min()
max_val = Features.max()
Features = (Features - min_val) / (max_val - min_val)

# 4) Skor fonksiyonu
def score(v1, v2):
    d = np.linalg.norm(v1 - v2)
    return 1 / (1 + d)

# 5) Genuine scores
genuine_scores = []

for person in range(100):
    for i in range(10):
        for j in range(i + 1, 10):
            s = score(Features[i, person], Features[j, person])
            genuine_scores.append(s)

print("Genuine score sayısı:", len(genuine_scores))

# 6) Imposter scores
imposter_scores = []

for t in range(10):
    for p1 in range(100):
        for p2 in range(p1 + 1, 100):
            s = score(Features[t, p1], Features[t, p2])
            imposter_scores.append(s)

print("Imposter score sayısı:", len(imposter_scores))

# 7) Histogram grafiği
plt.figure()
plt.hist(genuine_scores, bins=50, alpha=0.6, label="Genuine", density=True)
plt.hist(imposter_scores, bins=50, alpha=0.6, label="Imposter", density=True)
plt.xlabel("Score")
plt.ylabel("Density")
plt.title("Genuine vs Imposter Score Distribution")
plt.legend()
plt.grid(False)
plt.savefig("genuine_imposter.png", dpi=300, bbox_inches="tight")

# 8) Threshold, FAR, FRR
thresholds = np.linspace(0, 1, 1000)

far_list = []
frr_list = []

for th in thresholds:
    far = np.mean(np.array(imposter_scores) >= th)
    frr = np.mean(np.array(genuine_scores) < th)

    far_list.append(far)
    frr_list.append(frr)

# 9) EER hesapla
far_arr = np.array(far_list)
frr_arr = np.array(frr_list)

idx = np.argmin(np.abs(far_arr - frr_arr))
eer_threshold = thresholds[idx]
eer = (far_arr[idx] + frr_arr[idx]) / 2

print("EER:", eer)
print("EER threshold:", eer_threshold)

# 10) FAR-FRR grafiği
plt.figure()
plt.plot(thresholds, far_list, label="FAR")
plt.plot(thresholds, frr_list, label="FRR")
plt.scatter(eer_threshold, eer, label=f"EER = {eer:.4f}")
plt.xlabel("Threshold")
plt.ylabel("Error Rate")
plt.title("FAR ve FRR vs Threshold")
plt.legend()
plt.grid(False)
plt.savefig("far_frr_threshold.png", dpi=300, bbox_inches="tight")

# 11) FRR vs FAR grafiği
plt.figure()
plt.plot(far_list, frr_list)
plt.xlabel("FAR")
plt.ylabel("FRR")
plt.title("FRR vs FAR")
plt.grid(False)
plt.savefig("frr_vs_far.png", dpi=300, bbox_inches="tight")

plt.show()

    
