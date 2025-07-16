# 🎮 Player Engagement Predictor 🚀

![Streamlit Badge](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)

An advanced AI-powered web app built with **Streamlit** to **predict player engagement levels** based on in-game statistics and demographics. The app features a sleek, modern UI with support for both light and dark themes to enhance user experience.

---

## 🔗 Live Demo

Try it out now!

**▶️ [Launch the App](https://gba-optical.streamlit.app/)**

---

## ✨ Features

* 🧠 **AI-Powered Predictions**
  Uses a trained machine learning model to classify player engagement as **High**, **Medium**, or **Low**.

* 🎨 **Stunning UI/UX**
  Designed with custom CSS and Streamlit components for a premium interactive experience.

* 🌗 **Dark & Light Mode**
  Switch between a futuristic **Dark Mode** and a clean **Light Mode**.

* 📊 **Detailed Prediction Probabilities**
  Displays prediction confidence scores to provide deeper insight.

* 💡 **Actionable Insights**
  Auto-generated tips based on the prediction results and user input.

* 🛠️ **Debug Mode**
  Expandable section for technical users to view raw feature data and model expectations.

* 📱 **Mobile Responsive**
  Fully adaptive design for desktops, tablets, and mobile phones.

---

## 📸 Screenshots

![Dark Mode Screenshot](https://github.com/user-attachments/assets/85f91e76-4c7a-4773-a6b2-0386643845be)

---

## 🛠️ How It Works

1. **User Input**
   The user fills in in-game stats and demographics via an intuitive web form.

2. **Feature Engineering**
   Raw input is processed to include:

   * One-hot encoding (for `Location`, `GameGenre`, `GameDifficulty`)
   * Derived metrics (e.g., `EngagementPerSession`, `IntensityScore`)

3. **Data Scaling**
   Numerical features are scaled using a pre-trained `StandardScaler`.

4. **Prediction**
   The preprocessed data is fed into the trained model (`OP_model.pkl`) to predict engagement.

5. **Results Displayed**
   Results (category, probabilities, and insights) are shown using dynamic, interactive UI cards.

---

## 💻 Tech Stack

* **Language**: Python
* **Frontend/Framework**: Streamlit
* **Machine Learning**: Scikit-learn
* **Data Manipulation**: Pandas, NumPy
* **Model Storage**: Joblib

---

## ⚙️ Local Setup

### 1. Prerequisites

* Python 3.8+
* `pip` package manager

### 2. Clone the Repo

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 3. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Requirements

Create a `requirements.txt` with:

```txt
streamlit
pandas
numpy
scikit-learn
joblib
```

Then install:

```bash
pip install -r requirements.txt
```

### 5. Add Model Files

Ensure the following pre-trained files are in your project directory:

* `OP_model.pkl`
* `OP_scaler.pkl`
* `numeric_features.pkl`

> 🔁 Update file paths in `player.py` from absolute to relative like this:

```python
import joblib

model = joblib.load('OP_model.pkl')
scaler = joblib.load('OP_scaler.pkl')
numeric_features = joblib.load('numeric_features.pkl')
```

### 6. Run the App

```bash
streamlit run player.py
```

---

## 📁 Project Structure

```
.
├── OP_model.pkl              # Trained ML model
├── OP_scaler.pkl             # Fitted scaler
├── numeric_features.pkl      # Feature list
├── player.py                 # Main Streamlit app script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

---

## 🤝 Contributing

Contributions are welcome!
To get started:

1. **Fork** the repository
2. Create a new branch:
   `git checkout -b feature/YourFeatureName`
3. Make changes & commit:
   `git commit -m 'Add feature'`
4. Push to the branch:
   `git push origin feature/YourFeatureName`
5. Open a **Pull Request**

---

## 📄 License

Licensed under the **MIT License**.
See the [LICENSE](./LICENSE) file for details.

---
