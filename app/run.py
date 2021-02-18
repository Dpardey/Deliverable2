import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    
    # First graph
    genre_counts = round(df.groupby('genre').count()['message'] / df.shape[0] ,1) * 100
    genre_names = list(genre_counts.index)
    
    # Second graph
    y = df.drop(columns = ['id','message','original', 'genre','related'])
    categories_dist = round(y.sum(axis = 0) / y.shape[0],2) * 100
    top_10_categories = categories_dist.sort_values(ascending = False)[:-10]
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    width = 0.2,
                    #orientation = 'h',
                    opacity = 0.4
                )
            ],

            'layout': {
                #'height': '1024',
                'title': 'Distribution of the messages topics',
                'yaxis': {
                    'title': "Percentage over all data"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
        
        ,
        
        {
            'data': [
                Bar(
                    x = top_10_categories,
                    y = top_10_categories.index,
                    width = 1,
                    orientation = 'h',
                    opacity = 0.4                    
                )
            ],

            'layout': {
                #'height': '1024',
                'title': 'Categories distribution',
                'yaxis': {
                    'title': ""
                },
                'xaxis': {
                    'title': "Percentage over all the messages"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
