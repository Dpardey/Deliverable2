# Disaster Response Pipeline Project

### Summary

After a disaster had occurred, people started posting tweets or messages in the internet regarding the situation in a attempt to inform either friend or authorities.
Authorities, following a disaster, receive thousand of messages from the community either seeking for information or notifing of the disaster. 

¿How usefull it would be if the competent authorities have to power to extract valious information from all this data? It would result in a shorter response-time or a more integrated response across all the rescue teams. Ultimately, the goal is to prevent even more casualties that may have already occurred. 
The goal of this analysis is essentially that, identify text messages following disaster in order to identify the most valuable ones for the response teams.

This project was created within a single pipeline. Starting with the back-end (loading data, cleaning, ...) passing through an ML algorithm that learn from the data and endeed it in a front-end web app. 

### Files in the repository

1. App

 - template
   - go.html      *# classification result page of web app*
   - master.html # main page of web app
 - run.py # Flask file that runs app

2. Data

- Disaster_messages.csv   *# Database containing all the messages or tweets following a natural disaster*
   - **ID** type(*int*): ID of the message
   - **Message** type(*str*): The translated-to-english message
   - **Original** type(*str*): Message in the original language it was written
   - **Genre** type(*str*): Topic of the message

- Disaster_categories.csv   *#Database with all the pre-labels messages.*

    - **ID** type(*int*): ID ot the message
    - **Categories** type(*str*): Within this columns reside 35 categories marked one if the message are related to the category and zero otherwise 

- Process_data.py  *# Python script with the goal of cleaning the data*

- InsertDatabaseName.db  *# Database to save clean data to*

3. Models

- train_classifier.py   *# Python script that train and export the model*

- classifier.pkl   *# Exported model*

4. README.md

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
