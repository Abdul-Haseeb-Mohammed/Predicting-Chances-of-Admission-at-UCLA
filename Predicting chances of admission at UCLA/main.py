import pickle
import warnings
warnings.filterwarnings("ignore")
from src.data.make_dataset import load_data, clean_dataset
from src.features.build_features import prepare_data,train_test_splitter, standardize
from src.models.train_model import train_MLP, gridSearchCV
from src.models.predict_model import run_and_evaluate_model
from src.visualization.visualize import scatter_plot, loss_curve

if __name__ == "__main__":
    # Load the dataset
    data_path = "Predicting chances of admission at UCLA/data/raw/Admission.xls"
    df = load_data(data_path)

    #cleaning the data
    cleaned_dataset = clean_dataset(df.copy())
    """In the above dataset, the target variable is Admit_Chance
    To make this a classification task, let's convert the target variable into a categorical variable by using a threshold of 80%
    We are assuming that if Admit_Chance is more than 80% then Admit would be 1 (i.e. yes) otherwise it would be 0 (i.e. no)"""
    
    """Observations: The average GRE score of students applying for UCLA is ~316 out of 340. Some students scored full marks on GRE.
    The average TOEFL score of students applying for UCLA is ~107 out of 120. Some students scored full marks on TOEFL.
    There are students with all kinds of ratings for bachelor's University, SOP, and LOR - ratings ranging from 1 to 5.
    The average CGPA of students applying for UCLA is 8.57.
    Majority of students (~56%) have research experience.
    As per our assumption, on average 28.4% of students would get admission to UCLA."""
    
    print("\n\nVisualize the dataset to see some patterns:")
    scatter_plot(df,"Predicting chances of admission at UCLA/src/visualization/Scatter_Plot1.png")
    
    cleaned_dataset = prepare_data(cleaned_dataset)
    
    #Using standardization to scale the data
    x_scaled = standardize(cleaned_dataset.drop('Admit_Chance',axis=1))
    
    print(cleaned_dataset['Admit_Chance'].value_counts())
    # Split the dataset into train test
    X_train, X_test, y_train, y_test = train_test_splitter(x_scaled, cleaned_dataset['Admit_Chance'])
    
    #Train the Linear regression model
    MLP = train_MLP(X_train, y_train)

    # Save the trained model
    with open('Predicting chances of admission at UCLA/models/MLP_classifier.pkl', 'wb') as f:
        pickle.dump(MLP, f)

    MLP_train_mae, MLP_test_mae, MLP_train_cf, MLP_test_cf = run_and_evaluate_model(MLP, X_train, X_test, y_train, y_test)
    print('MLP Classifier Train error is', MLP_train_mae)
    print('MLP Classifier Test error is', MLP_test_mae)
    print("MLP Classifier Confusion Matrix:", MLP_train_cf)
    print("MLP Classifier Confusion Matrix:", MLP_test_cf)
    
    # Plotting loss curve
    loss_values = MLP.loss_curve_
    loss_curve(loss_values,"Predicting chances of admission at UCLA/src/visualization/loss_curve.png")
    
    print("Using Grid Search CV = 10 to find good parameters:")
    grid = gridSearchCV(MLP, x_scaled, cleaned_dataset['Admit_Chance'],10)
    
    print("Best parameters found using grid search:",grid.best_params_)
    print("Best Score:",grid.best_score_)
    
    # Save the best model to a file
    best_logistic_regression_model_filename = 'best_model_MLP.pkl'
    with open("Predicting chances of admission at UCLA/models/" + best_logistic_regression_model_filename, 'wb') as file:
        pickle.dump(grid.best_estimator_, file)
    print(f"Saved MLP best model as '{best_logistic_regression_model_filename}'")
    
    MLP_train_mae, MLP_test_mae, MLP_train_cf, MLP_test_cf = run_and_evaluate_model(grid.best_estimator_, X_train, X_test, y_train, y_test)
    print('Best MLP Classifier Train error is', MLP_train_mae)
    print('Best MLP Classifier Test error is', MLP_test_mae)
    print("Best MLP Classifier Confusion Matrix:", MLP_train_cf)
    print("Best MLP Classifier Confusion Matrix:", MLP_test_cf)
    
    