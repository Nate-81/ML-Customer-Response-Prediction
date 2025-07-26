# Customer Response to Insurance Prediction - XGBoost Project

A machine learning project by Anish Khanijaur (6480492) and Nathan Sakchiraphong (6480142)

## Project Overview

This project aims to predict which customers will respond positively to an automobile insurance offer using XGBoost classification. The project was originally developed in Google Colab and has been adapted to run locally.

## Problem Statement

The objective is to predict customer response to automobile insurance offers based on various demographic and policy-related features. This is a binary classification problem where:
- 0 = Customer will not respond positively
- 1 = Customer will respond positively

## Dataset

The dataset comes from a Kaggle competition: [Playground Series S4E7](https://www.kaggle.com/competitions/playground-series-s4e7/data)

### Data Setup

To run this project locally, you need to download the following files from the Kaggle competition and place them in the `./data/` directory:

1. `train.csv` - Training dataset with customer features and responses
2. `test.csv` - Test dataset for making predictions
3. `sample_submission.csv` - Sample submission format (already included)

### Features

The dataset includes the following features:
- **Age**: Age of the customer
- **Gender**: Gender of the customer
- **Region_Code**: Code for the region of the customer
- **Previously_Insured**: Whether the customer already has vehicle insurance
- **Vehicle_Age**: Age of the vehicle
- **Vehicle_Damage**: Whether the customer's vehicle was previously damaged
- **Annual_Premium**: The amount customer needs to pay as premium in the year
- **Policy_Sales_Channel**: Anonymized code for the channel of outreaching to the customer
- **Vintage**: Number of days the customer has been associated with the company
- **Response**: Target variable (1 if customer is interested, 0 otherwise)

## Project Structure

```
XGBoost-Project/
│
├── MlProject_Final.ipynb    # Main notebook with complete analysis
├── MLProject.ipynb          # Additional notebook (if needed)
├── README.md               # This file
└── data/
    ├── train.csv           # Training data (download from Kaggle)
    ├── test.csv            # Test data (download from Kaggle)
    └── sample_submission.csv # Sample submission format
```

## Key Features of the Analysis

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis of numerical and categorical features
- Correlation analysis
- Class imbalance investigation

### 2. Data Preprocessing
- Data type optimization for memory efficiency
- Feature encoding (gender, vehicle damage, vehicle age)
- Log transformation of annual premium
- Handling categorical variables with high cardinality

### 3. Class Imbalance Handling
The dataset is highly imbalanced (~8:1 ratio). The project explores multiple approaches:
- **Undersampling**: Reducing majority class samples
- **Class weights**: Using XGBoost's `scale_pos_weight` parameter
- **Combined approach**: Undersampling + class weights for optimal performance

### 4. Model Development
- XGBoost classifier with categorical feature support
- Custom evaluation metrics focusing on business objectives
- Emphasis on recall for positive class (identifying potential customers)

### 5. Business-Focused Evaluation
The model prioritizes **recall for the positive class** over traditional accuracy metrics because:
- False positives (marketing to uninterested customers) have low cost
- False negatives (missing interested customers) have high opportunity cost
- 94% recall achieved for identifying potential customers
- 99% precision for identifying customers who won't be interested

## Key Results

- **Final Model Configuration**: Undersampling (1:1 ratio) + Positive class weight (1.1)
- **Recall for Positive Class**: 94% (identifies 94% of potential customers)
- **Precision for Negative Class**: 99% (correctly identifies 99% of non-interested customers)
- **Business Impact**: Optimal balance between customer acquisition and marketing efficiency

## Model Performance Rationale

The chosen configuration prioritizes **recall for the positive class** because:

1. **Low Cost of False Positives**: Marketing to uninterested customers only costs advertising expenses
2. **High Cost of False Negatives**: Missing potential customers means lost revenue opportunities
3. **Customer Satisfaction**: Avoiding over-marketing to uninterested customers
4. **Business Efficiency**: 94% customer identification rate with minimal false advertising

## Technical Highlights

- **Memory Optimization**: Used Polars for efficient data handling and type optimization
- **Categorical Handling**: Leveraged XGBoost's built-in categorical feature support
- **Custom Evaluation**: Developed business-specific evaluation metrics
- **Feature Engineering**: Applied domain-appropriate transformations
- **Cross-validation**: Proper train-test splitting with stratification

## Future Improvements

1. **Hyperparameter Tuning**: Grid search or Bayesian optimization for XGBoost parameters
2. **Feature Engineering**: Creating interaction features or polynomial features
3. **Ensemble Methods**: Combining multiple models for improved performance
4. **Advanced Sampling**: SMOTE or other sophisticated sampling techniques
5. **Feature Selection**: Recursive feature elimination or feature importance analysis

## Authors

- **Anish Khanijaur** (6480492)
- **Nathan Sakchiraphong** (6480142)

