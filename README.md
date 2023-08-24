# End-to-End Credit Card Fraud Detection Project

## Overview

This repository contains an end-to-end Credit Card Fraud Detection project that showcases proficiency in Machine Learning algorithms, Exploratory Data Analysis (EDA), and cloud deployment. The project's aim is to develop a robust system for identifying fraudulent credit card transactions from a given dataset.

## Highlights

- **Machine Learning Algorithms:** The project involves the utilization of various Machine Learning algorithms including Logistic Regression, Decision Trees, Random Forest, Support Vector Machine, and XG Boost. Different algorithms were tested to determine the most suitable one for fraud detection.

- **Exploratory Data Analysis (EDA):** In-depth EDA was conducted to gain insights into the dataset. This step was crucial for understanding data distribution, patterns, and potential anomalies that may be indicative of fraudulent transactions.

- **Data Preprocessing:** The dataset contained class imbalance and outliers, which were addressed using techniques like oversampling and SMOTETomek.

- **Cloud Deployment:** The project was taken to the cloud with deployment on AWS Beanstalk. The deployment process was orchestrated using AWS CodePipeline for automated, efficient deployment.

## Project Structure

The project is structured as follows:

- `data`: Contains the dataset used for the analysis.
- `notebooks`: Jupyter notebooks detailing the EDA, data preprocessing, and ML model development.
- `src`: Source code modules including preprocessing scripts, model implementations, and Flask application for the front-end.
- `static` and `templates`: Front-end components including HTML templates and CSS styling.
- `setup.py`: Module for setting up the project environment.

## Getting Started

1. Clone this repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Follow the notebooks in the `notebooks` directory to understand the analysis process and model development.
4. Explore the `src` directory for the modularized code segments.

## Deployment

The project is deployed on AWS Beanstalk using AWS CodePipeline for streamlined deployment. To replicate the deployment process, follow the steps outlined in `deployment_steps.md`.

## Acknowledgements

The project was developed as a part of [Shayan_Kumar]'s personal exploration of Machine Learning, Data Analysis, and Cloud Computing.

## License

This project is licensed under the [MIT License](LICENSE).
