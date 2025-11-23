# 🏭 GCT Gabès - AI-Powered Circular Economy Solution

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18.x-61DAFB?style=for-the-badge&logo=react&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FF6600?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Transforming Industrial Waste into Sustainable Construction Materials**

*B2B Hackathon 2024 - Environmental Innovation Track*

[View Demo](#-live-demo) • [Features](#-features) • [Installation](#-quick-start) • [Documentation](#-project-structure)

</div>

---

## 🌍 The Problem

**Groupe Chimique Tunisien (GCT)** in Gabès, Tunisia faces critical environmental challenges:

- 🏔️ **30+ million tons** of phosphogypsum waste accumulated
- 💨 **High SO₂ emissions** causing respiratory diseases
- 🌊 **Marine pollution** destroying Gulf of Gabès ecosystem  
- 🏥 **Health crisis**: Gabès has Tunisia's highest cancer rates

**The cost**: Environmental degradation, health emergencies, and unsustainable industrial practices.

---

## 💡 Our AI-Powered Solution

We propose a **circular economy approach** that transforms pollution into value:

```
┌─────────────────────────────────────────────────────┐
│  Phosphate Processing                               │
│  ↓                                                   │
│  SO₂ Emissions + Phosphogypsum Waste                │
│  ↓                                                   │
│  ╔════════════════════════════════════════╗          │
│  ║  AI-POWERED TRANSFORMATION SYSTEM      ║          │
│  ║  • SO₂ Capture & Treatment             ║          │
│  ║  • Quality Control (Computer Vision)   ║          │
│  ║  • Treatment Optimization (ML)         ║          │
│  ║  • Real-time Monitoring (Digital Twin) ║          │
│  ╚════════════════════════════════════════╝          │
│  ↓                                                   │
│  High-Quality Eco-Bricks 🧱                         │
└─────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 1. 📊 **Digital Twin Dashboard** (React + Vite)
Real-time industrial monitoring system with ML-powered analytics:

- ✅ **SO₂ Capture Efficiency**: Live tracking with alerts
- 📈 **Production Metrics**: Gypsum flow, brick output, waste recycled
- 🤖 **ML Forecasting**: LSTM-based 24h pollution prediction
- 🏥 **Health Impact**: Disease reduction analysis for Gabès population
- 🌱 **Environmental KPIs**: CO₂ offset, waste recycling metrics

**Tech Stack**: React 18, Recharts, TailwindCSS, Vite

### 2. 🔬 **CV Quality Inspector** (PyTorch + MobileNetV3)
Computer vision system for phosphogypsum purity classification:

- 🎯 **Multi-Output Model**:
  - Whiteness score regression (0-100%)
  - Grade classification (A/B/C/D)
  - Defect detection (binary)
- ⚡ **Fast Inference**: ~80ms on CPU
- 📱 **Deployment Ready**: Streamlit demo, can run on Raspberry Pi
- 🎓 **Training Results**: 100% accuracy on synthetic data

**Tech Stack**: PyTorch, MobileNetV3, OpenCV, Streamlit

### 3. 🧪 **ML Treatment Optimizer** (XGBoost)
Predicts optimal treatment methods for brick strength:

- 📋 **Input Features**: Whiteness, impurity level, moisture, pH, particle size, temperature
- 🎯 **Predictions**: Standard, Enhanced Washing, Drying Required, Intensive Treatment
- 📊 **Performance**: 94.2% accuracy
- 💡 **Confidence Scoring**: Probability distribution for all treatment methods

**Tech Stack**: XGBoost, scikit-learn, pandas

---

## 🎬 Live Demo

### Dashboard
<img width="1229" height="656" alt="image" src="https://github.com/user-attachments/assets/d2cd2525-469d-49fa-950d-6b5d723c888d" />

*Real-time monitoring with live SO₂ tracking, ML forecasts, and health impact analysis*

### CV Quality Inspector
<img width="1226" height="671" alt="image" src="https://github.com/user-attachments/assets/d33555b2-50c8-4650-88a6-f8ba92dd3108" />
<img width="1220" height="371" alt="image" src="https://github.com/user-attachments/assets/aecf4e98-2f35-4691-9a18-a752376dcf30" />

*Upload phosphogypsum samples for instant quality grading*

### Training Results
<img width="4170" height="3565" alt="02_prediction_comparison" src="https://github.com/user-attachments/assets/3854142b-32ca-4eca-a2f3-43acdf6d4b15" />
<img width="3896" height="2498" alt="04_optimal_recipe_analysis" src="https://github.com/user-attachments/assets/faa2c86c-d127-4cac-acf7-4bd8dee7d34c" />


*96% validation accuracy on synthetic dataset*

---

## 📁 Project Structure

```
B2B-HACKTHON/
│
├── 📁 01_ML_Recipe_optimizer/           # XGBoost treatment optimizer
│   ├── ML prediction optimal treat.py   # Training script
│   ├── brick_treatment_model.pkl        # Trained model
│   ├── label_encoder.pkl                # Label encoder
│   ├── requirements.txt                 # Python dependencies
│   ├── output visualization/            # Training visualizations
│   └── README.md                        # Component documentation
│
├── 📁 02_CV_Quality_Inspector/          # Computer Vision quality control
│   ├── CV Model Training.py             # Model training
│   ├── streamlit demo app.py            # Streamlit web interface
│   ├── best_pg_model.pth                # Trained MobileNetV3 model
│   ├── training_results.png             # Training charts
│   ├── recipe_optimizer.pkl             # (legacy file)
│   ├── pg_dataset/                      # Generated synthetic images
│   │   ├── images/                      # 1000 PG sample images
│   │   └── metadata.json                # Labels and annotations
│   ├── requirements.txt                 # Python dependencies
│   └── README.md                        # Component documentation
│
├── 📁 03_Digital_Twin_Dashboard/        # React monitoring dashboard
│   ├── src/
│   │   ├── Dashboard.jsx                # Main dashboard component
│   │   ├── main.jsx                     # React entry point
│   │   └── index.css                    # Tailwind styles
│   ├── node_modules/                    # NPM packages (gitignored)
│   ├── index.html                       # HTML entry
│   ├── package.json                     # NPM dependencies
│   ├── package-lock.json                # Locked versions
│   ├── vite.config.js                   # Vite configuration
│   ├── tailwind.config.js               # Tailwind configuration
│   ├── postcss.config.js                # PostCSS configuration
│   ├── dashboard.jsx                    # Original source
│   └── README.md                        # Component documentation
│
├── 📄 README.md                         # This file
├── 📄 requirements.txt                  # Root Python dependencies
├── 📄 INSTALLATION.md                   # Setup guide
├── 📄 .gitignore                        # Git ignore rules
└── 📄 LICENSE                           # MIT License
```

---

## ⚡ Quick Start

### Prerequisites
```bash
Python 3.8+
Node.js 16+
Git
```

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/gct-gabes-ai-solution.git
cd gct-gabes-ai-solution
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Models (Optional - can use demo mode)

**CV Model:**
```bash
cd 02_CV_Quality_Inspector
python "CV Model Training.py"
```

**ML Optimizer:**
```bash
cd 01_ML_Recipe_optimizer
python "ML prediction optimal treat.py"
```

### 4. Run Applications

**Dashboard:**
```bash
cd 03_Digital_Twin_Dashboard
npm install
npm run dev
# Opens at http://localhost:3000
```

**CV Inspector:**
```bash
cd 02_CV_Quality_Inspector
streamlit run "streamlit demo app.py"
# Opens at http://localhost:8501
```

---

## 📊 Performance Metrics

| Component | Metric | Value |
|-----------|--------|-------|
| **CV Inspector** | Grade Accuracy | 100% (synthetic) |
| **CV Inspector** | Inference Time | ~80ms CPU |
| **CV Inspector** | Model Size | ~10MB |
| **ML Optimizer** | Treatment Accuracy | 94.2% |
| **ML Optimizer** | Training Time | <1 min |
| **Dashboard** | Update Frequency | 2 seconds |
| **Dashboard** | Build Size | ~200MB |

---

## 🛠️ Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React 18, Vite, TailwindCSS, Recharts, Lucide Icons |
| **ML/AI** | PyTorch, XGBoost, scikit-learn, MobileNetV3 |
| **Computer Vision** | OpenCV, Pillow, torchvision |
| **Data Processing** | NumPy, pandas, matplotlib |
| **Deployment** | Streamlit, Vite dev server |

---

## 🎯 Key Innovations

1. **Multi-Task CV Model**: Single model predicts whiteness, grade, and defects simultaneously
2. **Real-Time Digital Twin**: Simulates industrial process with realistic patterns
3. **ML Treatment Optimization**: Reduces trial-and-error in brick production
4. **Health Impact Quantification**: Links pollution reduction to disease cases
5. **Circular Economy**: Transforms 100% of waste into valuable products

---

## 📈 Environmental Impact

### Projected Annual Benefits (at scale):
- 🌬️ **SO₂ Captured**: 12,000+ tons/year
- ♻️ **Waste Recycled**: 800+ tons/year  
- 🌍 **CO₂ Offset**: 28,000+ kg/year
- 🏥 **Health Improvement**: 63% reduction in respiratory cases

---

## 🚧 Roadmap

- [x] Digital Twin Dashboard with real-time monitoring
- [x] CV-based quality inspection system
- [x] ML treatment optimizer
- [x] Synthetic data generation pipeline
- [ ] Real dataset collection from GCT facility
- [ ] Mobile app for field inspections
- [ ] API backend (FastAPI) for production deployment
- [ ] Integration with PLC/SCADA systems
- [ ] Multi-language support (Arabic, French, English)
- [ ] Real-time camera feed integration

---

## 📚 Documentation

- [Installation Guide](./INSTALLATION.md) - Complete setup instructions
- [Component READMEs](./01_ML_Recipe_optimizer/README.md) - Detailed component docs
- [API Documentation](#) - Coming soon

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👥 Team

  **Hackathon Information**
- Project: B2B Hackathon 2024
- Challenge: GCT Gabès Environmental Solution

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **B2B Hackathon Organizers** for the opportunity
- **Open-source community** for amazing tools and libraries
- **Gabès residents** whose health inspired this solution

---

## 📧 Contact

For questions or collaboration opportunities:

- Email: faouziblibech8@gmail.com


---
<img width="915" height="676" alt="image" src="https://github.com/user-attachments/assets/81e703b7-48b5-4ac8-a36b-ce210c24baf0" />


---

<div align="center">

**Built for a cleaner, healthier Gabès**

⭐ Star this repo if you found it helpful!

</div>





