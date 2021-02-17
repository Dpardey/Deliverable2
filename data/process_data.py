# Libraries needed for reading and manipulating the data
import sys
import pandas as pd
from sqlalchemy import create_engine
engine = create_engine('sqlite://', echo=False)



def load_data(messages_filepath, categories_filepath):
    '''
    Function that load the data needed
    
    Parameters:
    messages_filepath (str) : String containing the messages filepath database
    categories_filepath (str) : String cointainng the categories filepath database
    
    Returns:
    df (DataFrame) : Dataframe resulting dataframe after merging the messages and categories database
    
    '''
    global df_categories
    global df_messages    
    global df
    
    ## Loading the data
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)
    
    # Merging the data
    df = pd.merge(df_categories, df_messages, on = 'id')
    
    return df


def clean_data(df):
    '''
    Function that cleans the data
    
    Parameters:
    df (DataFrame) : Dataframe after merging the data
    
    Returns:
    df (DataFrame) : A clean DataFrame
    
    '''
    
    global index
    global category_colnames
    
    ## Index needed for a later merge
    index = df_categories.id
    
    ## Create a dataframe of the 36 individual category columns
    categories = df_categories.categories.str.split(';', expand = True)
    category_colnames = categories.iloc[0,].str.slice(start = 0, stop = -2)
    
    ## Renaming the columns with the category names
    categories.columns = category_colnames
    
    ## Set each value to be the last character of the string
    for column in categories:
        categories[column] = categories[column].astype(str).str.slice(start = -1).astype(int)
    
    ## Replacing the columns in df with categories 
    categories.insert(0, 'id', list(index))    
    
    df = df.drop('categories', axis = 1)
    df = pd.merge(df, categories, on = 'id')
    
    ## Dropping any duplicate row
    df = df.drop_duplicates()
        
    return df


def save_data(df, database_filename):
    '''
    Funtion that save the dataframe into a sqlite database
    
    Parameters:
    df (DataFrame) : Dataframe after merging the data
    database_filename (str) : A string containing the destination of the database    
    
    '''
    
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('DisasterResponse', engine, index=False, if_exists = 'replace') 
     


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
