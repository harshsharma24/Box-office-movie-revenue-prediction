# Box-office-movie-revenue-prediction

Movie Revenue Prediction
This project focuses on predicting movie revenues using various machine learning techniques, including feature engineering, handling missing data, and model optimization. The dataset contains information about movies such as their budget, genres, release dates, and more.

Project Overview
The goal is to build a machine learning model that predicts the revenue of a movie based on its attributes. The project includes:

Exploratory Data Analysis (EDA) to understand the dataset.
Feature engineering (e.g., extracting date features, handling categorical data).
Model training using Random Forest and XGBoost.
Hyperparameter tuning with GridSearchCV.
Evaluation of the model using metrics like RMSE.
Features
Feature Engineering:

Extracted features from dates (e.g., release year, release month, release weekday).
One-hot encoding for categorical variables (e.g., genres, languages).
Handled missing values by imputing or retaining them for specific models like XGBoost.
Models Used:

Random Forest: An ensemble method using bagging.
XGBoost: A boosting method optimized for structured data.
Optimization:

Hyperparameter tuning using GridSearchCV for XGBoost to find the best configuration.

Ensure you have the train.csv and test.csv files in the project directory.
Modify the paths in the code if necessary.
Run the Notebook:

Open the Jupyter notebook:
bash
Copy code
jupyter notebook "Harsh's_Movie_revenue_prediction.ipynb"
Follow the steps in the notebook to execute the pipeline.
Train the Model:

The notebook trains the Random Forest and XGBoost models and evaluates their performance using RMSE.
Make Predictions:

After training, the notebook generates predictions on the test dataset and saves the results.
Folder Structure
graphql


movie-revenue-prediction/
│
├── Harsh's_Movie_revenue_prediction.ipynb  # Main Jupyter notebook
├── train.csv                               # Training dataset
├── test.csv                                # Test dataset
├── requirements.txt                        # Dependencies
├── README.md                               # Project documentation

Requirements
Python 3.8 or above
Libraries:
pandas
numpy
scikit-learn
xgboost
matplotlib (for optional visualizations)
Results

Baseline RMSE (Mean Prediction): 3.120879515582972
Random Forest RMSE: ~0.87
Mean Squared Error (RMSE) on Training Data with XGBoost: 0.5695151501432907
Root Mean Squared Error (RMSE) on Training Data: 1.9290674400113856 (with default parameters; improved with GridSearchCV)

The project demonstrates how feature engineering and hyperparameter tuning can significantly improve model performance.

Future Enhancements
Experiment with additional features (e.g., cast and crew embeddings, advanced NLP on movie overviews).
Implement more ensemble techniques like stacking.
Explore deep learning models for potential performance gains.
Contributing
Contributions are welcome! Please fork the repository and create a pull request for review.

License
This project is licensed under the MIT License.

Acknowledgments
The dataset used in this project is sourced from Kaggle's TMDB Box Office Prediction dataset.
Thanks to open-source contributors for libraries like Scikit-Learn and XGBoost.