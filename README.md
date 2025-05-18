# Understanding & Predicting the Consumer Confidence Index (CCI) Using Key Macroeconomic Indicators

### Please note that this README.md file provides a brief summary and description of the project. 

### Please see the full Project Markdown File at this link: [Markdown Link](https://github.com/FujiPy/CCI-and-Key-Economic-Indicators/blob/main/CCIMarkdown.md)

**Author**: Max Fujimori

**Contact:** fujimorm@lafayette.edu

**Institution**: Lafayette College '27 | Economics & Data Science  


---

## 🔗 Project Links

- [Colab Notebook](https://colab.research.google.com/drive/19OQg3i31eW9riXzYHS99ohoML-Kr_TGU#scrollTo=Ix61nyLot8-F&uniqifier=1)
- [Video Presentation]()
- [Full Project Markdown File](https://github.com/FujiPy/CCI-and-Key-Economic-Indicators/blob/main/CCIMarkdown.md)
- [Kaggle Data Set](https://www.kaggle.com/datasets/sagarvarandekar/macroeconomic-factors-affecting-us-housing-prices/data)
- [ReadME .io Webpage](https://fujipy.github.io/CCI-and-Key-Economic-Indicators/)

---

## Project Overview

This project explores the relationship between the **Consumer Confidence Index (CCI)** and key **macroeconomic indicators** such as unemployment, inflation, GDP growth, mortgage interest rates, and household income. There are two main objectives:

1. **Analyze how current macroeconomic conditions shape consumer sentiment** using regression models.
2. **Evaluate whether CCI can serve as a leading indicator** to forecast future economic outcomes.

Using machine learning models including **polynomial regression**, **ridge regression**, and **random forests**, I quantified both coincident and forward-looking relationships between economic fundamentals and CCI. I also visualized variable trends over time and examined feature importance across models.

---

## Key Findings

- **Polynomial Ridge Regression** achieved an R² of **0.91**, indicating strong alignment between macroeconomic inputs and CCI.
- **Random Forest Regression** outperformed linear models with an R² of **0.95** and RMSE of **6.02**, showing high predictive accuracy for current sentiment.
- When treating **CCI as a 6 month leading indicator**, it forecasted **housing price levels** and **household income** moderately well (R² = 0.70 and 0.56), but had **low predictive power** for policy-driven variables like inflation or mortgage rates.
- Top predictive variables included **unemployment rate**, **home prices**, and **household income**, reflecting real economic conditions most visible to consumers.

---

## Sample Visuals

### Feature Correlation Matrix 
![png](CCIMarkdown_files/CCIMarkdown_37_0.png)
### Random Forest Regression Model with CCI as Target Scatter Plot
![png](CCIMarkdown_files/CCIMarkdown_61_1.png)

---

## Business & Policy Application

This framework offers real value to **economic policymakers and analysts**, enabling data-driven understanding of how macro conditions influence public sentiment. It also supports **early warning systems** by examining sentiment's ability to forecast future economic realities, especially in consumer-driven sectors like housing and retail.

---

## Software Used

- **Google Colab** (EDA and Machine Learning Models)
- **R-Studio** (Initial Markdown Development)
- **GitHub** (Upload, Versioning, & documentation)

---

##  File Structure

Colab Repository Structure:

├── CCIMarkdown_files    #Markdown Images

├── CCIMarkdown.md       #Full Project File in markdown format with code, notes, descriptions & links

├── US_MACRO_DATA.csv    # Raw Dataset

└── README.md            # This READ ME File 



