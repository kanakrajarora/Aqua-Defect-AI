# AquaDefect AI

**AquaDefect AI** is an AI-powered **water tank defect detection** application built with **Streamlit** and **YOLOv8n**.  
It enables real-time detection of multiple tank defects using pre-trained custom models on a proprietary dataset.

---

## 🚀 Features
- Simple **Streamlit web interface** for defect detection.
- Supports multiple defect types in water tanks.
- Uses **YOLOv8n** models fine-tuned on a custom image dataset.
- Modular design for easy integration of new detection types.

---

## 🧾 Supported Defects

| Defect Name | Detection Function |
|-------------|--------------------|
| Roughness | `Detect_Roughness` |
| Lumps | `detect_lumps` |
| Black Shade | `Detect_shade` |
| Sunlight | `Detect_sunlight` |
| Potholes | `Detect_potholes` |
| Rust | `Detect_non_blue_regions` |
| Diameter | `Detect_diameter` |
| Material-Mixing | `Detect_material_mix` |
| Joint-Cut | `joint_cut` |
| Mould-Joint-Mismatch | `mould_joint_mismatch` |
| Black Lining | `black_lining` |
| Burning Marks | `burn_white_mark_detection` |
| Damage | `detect_damage` |
| Improper Weld Finishing | `improper_weld_finishing` |
| Blow Hole | `detect_blowhole` |
| Blow Hole due to Contamination | `detect_blowhole_contamination` |

---

## 📂 Project Structure

```
AquaDefect-AI/
│
├── blowhole.pt                 # YOLOv8n model for Blow Hole detection
├── blowhole_contamination.pt   # Pretrained YOLOv8n model for Contamination Blow Hole detection
├── damage.pt                   # Pretrained YOLOv8n model for Damage detection
├── weld.pt                     # Pretrained YOLOv8n model for Weld defect detection
├── streamlit_app.py            # Main Streamlit application
├── utils.py                    # Utility functions for detection
└── README.md                   # Project documentation
```

---

## 🛠 Installation

1. **Clone the repository**
```bash
git clone https://github.com/kanakrajarora/Aqua-Defect-AI.git
cd Aqua-Defect-AI
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**
```bash
streamlit run streamlit_app.py
```

---

##  Requirements
- Python 3.8+
- Streamlit
- Ultralytics YOLOv8
- OpenCV
- NumPy

---

##  Model Training
The YOLOv8n models (`.pt` files) were **trained on a custom image dataset** specifically curated for water tank defect detection.  
Each defect type has its own fine-tuned model for high accuracy.


---

##  License
This project is licensed under the MIT License.
