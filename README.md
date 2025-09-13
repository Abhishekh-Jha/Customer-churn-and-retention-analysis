# Customer-churn-and-retention-analysis

Project Overview:

This project is a comprehensive analysis of customer churn for a telecom company. Using a data-driven approach, I performed a detailed investigation to identify the key factors that influence customer attrition. The goal was to build a predictive model to identify high-risk customers and provide actionable business recommendations to improve customer retention.

Data Source:

The analysis is based on the "Telco Customer Churn" dataset, which includes customer demographic information, service subscriptions, and billing details.

Methodology:

The project follows a standard data science workflow, documented across four Jupyter notebooks:

01_data_load_and_EDA.ipynb: Data Loading & Exploratory Data Analysis

  - Loaded the raw dataset and performed initial data cleaning, including handling missing values in the TotalCharges column.

  - Analyzed the distribution of customer churn, identifying a significant class imbalance with ~26.6% of customers churning.

   - Explored the relationships between churn and various features through visualizations, uncovering initial insights on key drivers like contract type, tenure, and internet service.

02_Preprocessing_and_featureEngg.ipynb: Preprocessing & Feature Engineering

  - Prepared the data for machine learning models by encoding categorical variables (e.g., gender, Contract).

  - Scaled numerical features (tenure, MonthlyCharges, TotalCharges) to ensure all variables were on a uniform scale.

  - Engineered new, meaningful features such as TenureGroup, TotalServices, and TotalRevenue to enhance model performance and provide richer insights.

03_modeling.ipynb: Machine Learning Models

  - Split the dataset into training and testing sets, preserving the class imbalance using a stratified split.

  - Trained and evaluated a baseline Logistic Regression model.

  - Developed and tested additional models, including a Decision Tree and a Random Forest classifier.

  - Used key metrics like accuracy, precision, recall, and F1 score, along with a confusion matrix, to assess each model's performance.

04_Model_Comparison_&_Conclusion.ipynb: Model Comparison & Key Insights

  - Compared the performance of all three models, concluding that Logistic Regression provided the best balance of predictive power and interpretability with an F1 score of 0.61 and an ROC-AUC of 0.84.

  - Analyzed and visualized feature importance from the top-performing models to validate key churn drivers.

Key Findings & Business Recommendations:

   My analysis identified the following critical drivers of customer churn:

  - Contract Type: Customers on a month-to-month contract are significantly more likely to churn compared to those with longer-term contracts.

  - Internet Service: Fiber optic users have a much higher churn rate, indicating potential service dissatisfaction.

  - Customer Lifecycle: Lower tenure and lower TotalCharges are strong indicators of churn, suggesting that new customers are a high-risk group.

  - Payment & Support: The "Electronic check" payment method and the lack of OnlineSecurity or TechSupport services are highly correlated with churn.

  - Based on these findings, I recommend the following for the business:

  - Retention Offers: Implement targeted retention campaigns with discounts or incentives specifically for customers on month-to-month contracts.

  - Service Improvement: Investigate and improve the fiber optic service to address potential performance or support issues.

  - Promote Stability: Encourage customers to switch to 1-year or 2-year contracts by highlighting the long-term value and stability they offer.

  - Enhance Onboarding: Focus on improving the onboarding process for new customers to increase their engagement and tenure.
