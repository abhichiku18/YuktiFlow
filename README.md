# 🌾 YuktiFlow — Intelligent Business Insight Dashboard

**YuktiFlow** is an AI-powered web application that helps businesses forecast sales, predict demand, detect stockout risks, and classify products as fast- or slow-moving — all from a single interactive dashboard.

🔗 **Live Demo:** [https://yuktiflow-2ud9.onrender.com](https://yuktiflow-2ud9.onrender.com)

---

## 🚀 Features

- 📈 **Sales Forecasting (Prophet Model):** Predicts future sales trends using Facebook Prophet.  
- 📊 **Demand Prediction (Random Forest):** Estimates upcoming demand based on product and region.  
- ⚠️ **Stockout Risk Detection (Logistic Regression):** Identifies products with high risk of going out of stock.  
- 🌀 **Product Speed Classification (K-Means):** Clusters products into fast- and slow-moving categories.  
- 📂 **CSV Upload:** Upload your own dataset and instantly visualize predictions.  
- 🌟 **Interactive Dashboard:** Beautiful UI built with HTML, CSS, and Flask for a modern analytical experience.  

---

## 🧩 Tech Stack

| Layer | Technologies Used |
|-------|-------------------|
| **Frontend** | HTML5, CSS3 (Custom Styling), JavaScript |
| **Backend** | Python (Flask Framework) |
| **Machine Learning Models** | Prophet, Random Forest, Logistic Regression, KMeans |
| **Libraries** | pandas, scikit-learn, matplotlib, joblib |
| **Deployment** | Render (Flask Web App) |

---

## 🧠 How It Works

YuktiFlow makes business forecasting and analytics simple, smart, and interactive. Here’s how it works step-by-step:

### 1️⃣ Upload Your CSV File
Start by uploading a `.csv` file containing your business or sales data.  
The file should include columns such as:
- **Date** → Date of sales activity  
- **Product** → Product name or ID  
- **Area** → Sales region or market  
- **Units_Sold** → Number of items sold  
- **Stockout_Risk (optional)** → Risk score or probability of product stockout  

---

### 2️⃣ Data Processing & Model Execution
Once uploaded, YuktiFlow automatically validates and preprocesses your data before passing it through multiple machine learning models:

- 🤖 **Prophet Model:** Performs **time-series forecasting** for the next 30 days.  
- 🌳 **Random Forest Model:** Predicts **demand levels** based on product, area, and stockout risk.  
- ⚡ **Logistic Regression Model:** Determines **stockout risk levels** (High/Low).  
- 🎯 **K-Means Clustering:** Classifies products as **Fast Moving** or **Slow Moving** based on historical performance.  

---

### 3️⃣ Interactive Results & Visualization
After computation, the system generates interactive insights on the dashboard:
- 📈 A **Sales Forecast Graph** showing predicted trends for the next 30 days.  
- 📋 A **Comprehensive Prediction Table** combining outputs from all models.  
- 💡 **Business Insights** highlighting which products are performing well, which need attention, and which are at risk of stockouts.  

---

### 4️⃣ Analyze & Export
- View all predictions directly within the dashboard.  
- Download the complete results as a `.csv` file for reporting or deeper analysis.  
- Use these insights to make **data-driven business decisions**, optimize inventory, and forecast sales performance efficiently.  

---

## 🧑‍💻 Author

**👨‍💻 Abhinav Kumar Chaudhary**  
App Developer | Machine Learning Enthusiast | Team Leader  

📧 [abhichiku2004@gmail.com](mailto:abhichiku2004@gmail.com)  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/abhichiku/)  
💻 [GitHub Profile](https://github.com/abhichiku18)

---

## 🏁 Future Enhancements

- 🌐 Integrate with live sales APIs (Google Sheets / ERP)  
- 📊 Add advanced dashboards with Plotly visualizations  
- 🧮 Enable deep learning forecasting models (LSTM)  
- 🔒 Add authentication for business users  

---

## 📜 License

This project is open-source under the **MIT License**.  
Feel free to fork, modify, and use it for learning or business analytics projects.  

---

⭐ **If you like this project, give it a star on GitHub!** ⭐
