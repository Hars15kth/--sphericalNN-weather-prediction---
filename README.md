#  Spherical Weather Prediction

This repository benchmarks baseline CNNs against SO(3)-equivariant spherical models for ERA5 500 hPa wind prediction. It is designed for reproducibility,While focused on climate modeling, the pipeline architecture and equivariant design principles are directly transferable to LLM infrastructure and RAG systems—especially in scenarios requiring spatial reasoning, rotation-aware embeddings, or robust cache generation across distributed inputs.
 

## ✅ Highlights

- Minimal, crash-proof PyTorch pipeline (Windows-native, no Docker)
- Spherical mesh construction with chunked KNN indexing
- Baseline and spherical cache generation with atomic `.npy` saves
- Visual comparison of wind fields and improvement heatmaps
- Restrictive licensing for portfolio protection
- Modular design patterns applicable to LLM pre-processing and RAG embedding pipelines



---

##  Performance Summary

| Metric            | Baseline     | Spherical    | Δ (Spherical - Baseline) |
|-------------------|--------------|--------------|---------------------------|
| MSE               | 0.5319       | 0.5257       | ✅ -1.2%                  |
| MAE               | 0.5533       | 0.5471       | ✅ -1.1%                  |
| RMSE              | 0.7293       | 0.7250       | ✅ -0.6%                  |
| Angle Error       | 0.4181       | 0.4336       | ❌ +3.7%                  |
| Mag MAE           | 0.5512       | 0.4546       | ✅ -17.5%                 |
| Divergence Abs    | 0.1841       | 0.1941       | ❌ +5.4%                  |
| MSE North         | 0.6267       | 0.6235       | ✅ -0.5%                  |
| MSE South         | 0.4370       | 0.4279       | ✅ -2.1%                  |
| Composite Score   | 0.6798       | 0.6676       | ✅ -1.8%                  |

> Spherical model wins on 6 of 9 metrics, with significant gains in magnitude MAE and southern hemisphere MSE. Despite a slight increase in angle error, overall vector fidelity and magnitude accuracy are improved.

---

##  Visual Output

![ERA5 Winds Comparison](results/plots/era5_winds_comparison.png)

This figure compares zonal (u) and meridional (v) wind components at 500 hPa across:

- **Truth**: ERA5 reanalysis data  
- **Baseline**: Standard CNN model  
- **Spherical**: SO(3)-equivariant spherical model  

The bottom row shows **improvement heatmaps**:
- **Red regions** indicate areas where the spherical model has lower absolute error than the baseline.
- **Blue regions** indicate areas where the baseline performs better.

> The spherical model shows clear improvement in mid-latitude and southern hemisphere wind fields, especially in magnitude and directional coherence.

---

##  Repo Structure

├── scripts/              # Preprocessing, cache building, training ├── models/               # Equivariant tensor product blocks ├── notebooks/            # Visual exploration and analysis ├── results/              # Plots and metrics for recruiter clarity ├── requirements.txt      # Minimal version-pinned dependencies ├── LICENSE               # Custom restrictive license └── README.md 



---

##  License

This repository is strictly licensed for personal portfolio use. Redistribution or commercial use is prohibited without written consent.

---



