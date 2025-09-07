# Patient Data Clustering and Analysis


## Project Overview

This project performs an in-depth exploratory data analysis (EDA) and applies unsupervised machine learning techniques to a patient health dataset. The primary goal is to identify natural groupings or "clusters" of patients based on their health metrics, such as blood pressure, cholesterol, and age. By understanding these clusters, we can uncover hidden patterns and segment patients into distinct profiles, which could be valuable for targeted healthcare interventions.

The analysis involves data cleaning, preprocessing, and the application of three different clustering algorithms: K-Means, Hierarchical Clustering, and DBSCAN. Finally, the trained K-Means model and the data scaler are saved for future use.

## Key Features

- Exploratory Data Analysis (EDA): Comprehensive analysis of the dataset's statistical properties and feature distributions.

- Data Preprocessing: Robust handling of missing values and feature scaling using StandardScaler to prepare the data for modeling.

- Clustering Analysis: Implementation and comparison of three popular clustering algorithms:

- K-Means: To partition patients into a predefined number of groups.

- Hierarchical Clustering: To build a hierarchy of clusters and understand nested groupings.

- DBSCAN: To identify density-based clusters and detect outliers.

- Model Persistence: The trained K-Means model and the data scaler are saved as .pkl files, allowing for easy reuse without retraining.

- Statistical Analysis: A chi-squared test was conducted to investigate the correlation between categorical variables (residence_type and heart_disease).

## Repository File Structure
```
.
├── patient_dataset.csv     # The raw patient dataset used for the analysis.
├── learning.ipynb          # Jupyter Notebook with detailed EDA and experimental clustering.
├── task.ipynb              # Jupyter Notebook focused on the final model training and saving process.
├── kmeans_model.pkl        # Saved/serialized K-Means clustering model object.
├── scaler.pkl              # Saved/serialized StandardScaler object for data preprocessing.
└── requirements.txt        # A list of Python libraries required to run the project.
```

## How To Use

To replicate this analysis on your local machine, please follow these steps:

### 1. Clone the repository:
```
git clone [https://github.com/your-username/patient-data-clustering.git](https://github.com/your-username/patient-data-clustering.git)
cd patient-data-clustering
```
### 2. Create a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install the required libraries:
```
pip install -r requirements.txt
```
### 4. Run the Jupyter Notebooks:
Launch Jupyter and open either learning.ipynb for the full exploratory journey or task.ipynb for the streamlined model creation process.
```
jupyter notebook
```

##  Summary of Findings

The analysis revealed several key insights into the patient data:

Four Distinct Patient Profiles: Both K-Means and Hierarchical clustering suggested the presence of four primary patient clusters:

At-Risk Group: Older patients with high blood pressure, high cholesterol, and a higher incidence of heart disease.

Younger, Healthier Group: The youngest cohort with normal vitals and low prevalence of chronic conditions.

Pre-Diabetic/Metabolic Syndrome Profile: Characterized by high plasma glucose and insulin levels.

Overweight Group: Defined by a high BMI but with otherwise moderate risk factors.

Outlier Detection: The DBSCAN algorithm identified a single large, dense cluster representing the "typical" patient, along with several outliers whose health profiles deviate significantly from the norm.

No Correlation Found: The chi-squared analysis showed no statistically significant correlation between a patient's residence type (Urban vs. Rural) and their likelihood of having heart disease (p-value = 0.46).

## Clustering Performance Evaluation

- Evaluating the performance of clustering algorithms is not as straightforward as with supervised learning, as there are no ground truth labels. Instead, intrinsic metrics are used to assess the quality of the clusters.

- Silhouette Score: Measures how well-separated the clusters are. A score closer to 1 indicates dense and well-separated clusters.

- Calinski-Harabasz Score: Also known as the Variance Ratio Criterion, it measures the ratio of between-cluster dispersion to within-cluster dispersion. A higher score is better.

- Davies-Bouldin Index: Measures the average similarity of each cluster with its most similar cluster. A lower score (closer to 0) is better.

## The performance of the final K-Means model (with k=4 clusters) is as follows:

```

 Method	       | Silhouette	 |  Davies-Bouldin	 |  Calinski-Harabasz
 KMeans	       |  0.179827	  |    1.951107	     |   269.881229
 Hierarchical	 |  0.179827	  |    1.951107	     |   269.881229
 DBSCAN       	|  0.175863	  |    4.298287	     |   132.919588


```


## Future Work
- Predictive Modeling: Build a supervised learning model to predict the likelihood of heart disease based on patient features.

- Deployment: Use the saved kmeans_model.pkl and scaler.pkl files to build a simple web application (using a framework like Gradio or Flask) that can assign a new patient to a cluster in real-time.

- Deeper Cluster Analysis: Perform a more granular analysis of the identified clusters to understand the specific combination of factors that define each patient profile.
