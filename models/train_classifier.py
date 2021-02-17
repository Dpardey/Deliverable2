# Libraries needed for reading and manipulating the data
import sys
import pandas as pd 
import numpy as np
from sqlalchemy import create_engine
engine = create_engine('sqlite://', echo=False)

# NLP packeages
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Transform, split, train ML algorithms
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

# This package enables to print statments in the middle of the screen
import shutil
columns = shutil.get_terminal_size().columns

# Saving the model
import pickle


def load_data(database_filepath):
    '''
    Function that import the data and store it in a sqlite database
    
    Parameters: 
    database_filepath (str) : Location of the sqlite database
    
    Returns:
    X (array) : Object with the messages within the database
    Y (DataFrame) : Object with the categories of the message
    category_names (list) : List with the category names 
    
    '''
    
    global df
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('DisasterResponse',engine)
    X = df.message.values
    Y = df.loc[:,~df.columns.isin(['id','message','original','genre'])]
    Y.loc[Y.related > 1, 'related'] = 1
    Y = Y.drop(columns = 'child_alone')
    category_names = [i for i in df.columns if i not in (['id','message','original','genre'])]
    
    return X, Y, category_names


def tokenize(text):
    '''
    
    Function that clean the messages object
    
    Parameters:
    text (array) : Object with the messages
    
    Returns:
    clean_tokens : Object containing the "clean" messages
    
    '''
    
    
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    
    for token in text:
        
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()

        clean_tokens = []
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)   
        
    return clean_tokens


def build_model():
    '''
    Function that create and boost the model performance
    
    Parameters:
    
    Returns:
    cv : Model
    
    '''
    
    stop_words = stopwords.words('english')
    
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect',  CountVectorizer(stop_words = stop_words, token_pattern = r'[^\w]', tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),

        ('svc', MultiOutputClassifier(LinearSVC()))
    ])
    
    parameters = {
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'svc__estimator__C': [0.1, 1, 10, 100]
    } 

    cv = GridSearchCV(pipeline, param_grid=parameters)
    #cv.fit(X_train, Y_train)
    
    #print('\nOverall best parameters: ', cv.best_params_)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function that prints out the metrics of categories
    
    Parameters:
    model: Model
    X_test (array) : Object containing the randomized proportion of the messages
    Y_test (DataFrame) : Object with the categories of the messages
    category_names: List with the category names
    
    Returns:
    
    '''
    
    prediction = model.predict(X_test)
    for i, category in enumerate(category_names):
        
        print('... {} ...'.format(category.upper()).center(columns))
        print(classification_report(Y_test[category], prediction[:,i]))


def save_model(model, model_filepath):
    '''
    Function that save the model
    
    Parameters:
    model: Model
    model_filepath (str): The model filepath
    
    '''
    
    pkl_filename = "classifier.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
