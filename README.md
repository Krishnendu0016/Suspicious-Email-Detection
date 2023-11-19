# Suspicious Email Detection App

This Streamlit app uses a Random Forest Classifier to predict whether an input email is suspicious (spam) or not. The model is trained on a dataset containing labeled spam and ham (non-spam) emails.

## Usage

1. Make sure you have the required dependencies installed. You can install them using the following command:

    ```bash
    pip install numpy pandas scikit-learn streamlit
    ```

2. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

3. Enter the email text in the provided text area.
4. Click the "Predict" button to see the prediction result.

## Files

- `app.py`: The main Python script containing the Streamlit app.
- `mail_data.csv`: CSV file containing email data used for training.
- `spam_ham_dataset.csv`: Additional CSV file with email data for training.
- `README.md`: This documentation file.

## Dependencies

- numpy
- pandas
- scikit-learn
- streamlit

## Model Training

The Random Forest Classifier is trained on a combination of the `mail_data.csv` and `spam_ham_dataset.csv` datasets. The data is pre-processed and transformed using TF-IDF vectorization.
