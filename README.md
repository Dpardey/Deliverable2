# Disaster Response Pipeline Project

### Summary

The goal of this project is to identify text messages following a natural disaster in order to identify the most important (valuable for the responde disaster professionals). 
This project was created within a single pipeline. Starting with the back-end (loading data, cleaning, ...) passing through an ML algorithm that learn from the data and endeed it in a front-end with a web app. 

### Databases

1. Disaster_messages.csv: Database containing all the messages or tweets following a natural disaster.

- ID (int): ID of the message
- Message (str): The translated-to-english message
- Original (str): Message in the original language it was written
- Genre (str): Topic of the message

  - Number of messages in the data: 26.248

2. Disaster_categories.csv: Database with all the pre-labels messages.

- ID (int): ID ot the message
- Categories (str): Within this columns reside 35 categories marked one if the message are related to the category and zero otherwise 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
