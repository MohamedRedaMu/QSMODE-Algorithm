# QSMODE-Algorithm
Q-Spline Multi-Operator Differential Evolution (QSMODE) for path planning

Official MATLAB implementation of the QSMODE algorithm.

## 📄 Paper

Mohamed Reda, Ahmed Onsy, Amira Y. Haikal, and Ali Ghanbari  
**A novel reinforcement learning-based multi-operator differential evolution with cubic spline for the path planning problem**  
Artificial Intelligence Review, Volume 58, Article 142, 2025  
DOI: https://doi.org/10.1007/s10462-025-11129-6

---

## Overview

QSMODE is a reinforcement learning-enhanced differential evolution algorithm for path planning. It combines:

- multi-operator differential evolution
- Q-learning
- cubic spline path smoothing
- adaptive archive and parameter memories
- local search for path refinement

According to the published paper, QSMODE and DQSMODE were developed under a unified framework and evaluated on the RP2B-24 benchmark, which contains 50 path-planning problems. The paper also reports practical validation on a ROS-based four-wheel differential drive robot. :contentReference[oaicite:1]{index=1}

---

## Main Contributions

- Reinforcement learning is used directly for path generation, not only for parameter tuning
- Q-table update is performed through:
  - random-action exploration
  - self-training from population-generated candidate paths
- Cubic spline interpolation improves path smoothness
- RP2B-24 benchmark provides a unified benchmark library for path planning

---

## 📂 Repository Structure

```text
QSMODE-Algorithm/
│
├── QSMODE_algorithm.m
├── CreateModel_RP2B24_Benchmark.m
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
└── .gitattributes
```


---

## ⚙️ Requirements

- MATLAB (recommended R2023a or later)
- Optimization Toolbox (`fmincon`)
- Statistics and Machine Learning Toolbox (`normrnd`)
- Benchmark helper files used by the current implementation

---

## ▶️ How to Run

Open MATLAB, set the current folder to the repository root, then run:

```matlab
[goalReached, GlobalBest, countFE, Q, replay_memory] = QSMODE_algorithm();
```

### Current default internal settings

```matlab
modelType = 4;
modelNum = 8;
boundaries.xmin = -10;
boundaries.xmax = 10;
boundaries.ymin = -10;
boundaries.ymax = 10;
fNo = 1;
```

The current implementation builds the benchmark model internally using:

```matlab
model = CreateModel(modelType, modelNum, boundaries, fNo);
```

---

## 🔧 Benchmark Notes

The published paper introduces the RP2B-24 benchmark library with 50 distinct path-planning problems and states that both QSMODE and DQSMODE are evaluated under this framework. The source code links for both the benchmark and QSMODE are explicitly provided on the publisher page. :contentReference[oaicite:2]{index=2}

Current benchmark-related settings in the code include:

- `modelType`
- `modelNum`
- `boundaries`
- `fNo`
- `gridResolution`
- `maxSteps`
- `maxCount`

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{reda2025novel,
  title   = {A novel reinforcement learning-based multi-operator differential evolution with cubic spline for the path planning problem},
  author  = {Reda, Mohamed and Onsy, Ahmed and Haikal, Amira Y. and Ghanbari, Ali},
  journal = {Artificial Intelligence Review},
  volume  = {58},
  number  = {5},
  pages   = {142},
  year    = {2025},
  publisher = {Springer},
  doi     = {10.1007/s10462-025-11129-6}
}
```

### APA
Reda, M., Onsy, A., Haikal, A. Y., & Ghanbari, A. (2025). A novel reinforcement learning-based multi-operator differential evolution with cubic spline for the path planning problem. *Artificial Intelligence Review, 58*(5), 142.

### Chicago
Reda, Mohamed, Ahmed Onsy, Amira Y. Haikal, and Ali Ghanbari. “A novel reinforcement learning-based multi-operator differential evolution with cubic spline for the path planning problem.” *Artificial Intelligence Review* 58, no. 5 (2025): 142.

---

## 📜 License

This project is released under the MIT License. See the `LICENSE` file for details.

---

## 📧 Contact

**Dr. Mohamed Reda**  
University of Central Lancashire, UK  
Mansoura University, Egypt

- 📩 Personal: [mohamed.reda.mu@gmail.com](mailto:mohamed.reda.mu@gmail.com)  
- 📩 Academic: [mohamed.reda@mans.edu.eg](mailto:mohamed.reda@mans.edu.eg)  

---

## 🌐 Academic Profiles

- 🧑‍🔬 ORCID: https://orcid.org/0000-0002-6865-1315
- 🎓 Google Scholar: https://scholar.google.com/citations?user=JmuB2qwAAAAJ
- 📊 Scopus: https://www.scopus.com/authid/detail.uri?authorId=57220204540
- 📚 Web of Science: https://www.webofscience.com/wos/author/record/3164983
- 🧾 SciProfiles: https://sciprofiles.com/profile/Mreda

---

## 🔗 Professional & Social Links

- 💼 LinkedIn: https://www.linkedin.com/in/mraf
- 🔬 ResearchGate: https://www.researchgate.net/profile/Mohamed-Reda-8
- 🎓 Academia: https://mansoura.academia.edu/MohamedRedaAboelfotohMohamed
- 📘 SciLit: https://www.scilit.net/scholars/12099081
- 🧮 MATLAB Central: https://uk.mathworks.com/matlabcentral/profile/authors/36082525
- ▶️ YouTube: https://youtube.com/@mredacs

---

## Related Repository

- RP2B24 Benchmark: https://github.com/MohamedRedaMu/RP2B24-Benchmark

---

## Acknowledgement

This repository accompanies the published QSMODE paper in *Artificial Intelligence Review*.
