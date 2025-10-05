\# 🌍 Spherical Weather Prediction



This repository benchmarks baseline CNNs against SO(3)-equivariant spherical models for ERA5 500 hPa wind prediction. It includes:



\-  Minimal, reproducible PyTorch pipeline

\-  Spherical mesh construction and KNN indexing

\-  Baseline and spherical cache generation

\-  Visual comparison of model outputs

\-  Strict licensing and portfolio-grade documentation



\##  Structure

&nbsp; 



├── scripts/              # Preprocessing, cache building, training ├── models/               # Equivariant tensor product blocks ├── notebooks/            # Visual exploration and analysis ├── results/              # Plots and metrics for recruiter clarity ├── requirements.txt      # Minimal version-pinned dependencies ├── LICENSE               # Custom restrictive license └── README.md  







\##  Visual Output



!\[ERA5 Winds Comparison](results/plots/era5\_winds\_comparison.png)



> Generated after multiple kernel crash recoveries and memory-safe plotting attempts.



\## License



This repository is strictly licensed for personal portfolio use. Redistribution or commercial use is prohibited.



\##  Author



Built by Harshwardhan — methodical, outcome-driven DL project focused on reproducibility, benchmarking, and authenticity.

