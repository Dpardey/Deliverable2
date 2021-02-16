# import libraries
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

## Pickle
import pickle


def load_data(database_filepath):
    
    global df
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('DisasterResponse',engine)
    X = df.message.values
    Y = df.loc[:,~df.columns.isin(['id','message','original','genre'])]
    
    return X, Y


def tokenize(text):
    
    ## Here we're imporing the stop words list, initializing the lemmatizer and creating 
    ## an empty list to append letter the clean sentences
    
    stop_words = stopwords.words('english')
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
    
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect',  CountVectorizer(stop_words = stop_words, token_pattern = r'[^\w]', tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),

        ('clf', MultiOutputClassifier(LinearSVC()))
    ])
    
    parameters = {
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    } 

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    categories = [i for i in df.columns if i not in (['id','message','original','genre'])]
    print(classification_report(X_test, cv.fit(X_test, Y_test), target_names=categories))


def save_model(model, model_filepath):
    
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