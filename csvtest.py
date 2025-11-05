import os
import re
import pandas as pd

question_dict = {}

def get_dataset_path():
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the CSV file
    csv_path = os.path.join(current_dir, "data", "food_waste_survey.csv")
    return csv_path

def extract_questions():
    df = pd.read_csv(get_dataset_path())
    pattern = r'Q\d+[A-Z]?\.'
    
    columns = df[df.columns[0]]
    print(re.findall(pattern, columns[3]))
    question_list = []


    for column in columns:
        if (isinstance(column, str)):
            question = re.findall(pattern, column)
            if ( len(re.findall(pattern, column)) != 0 and (not list_contains(question_list, column[0:4]))):
                sort_dict(question_list, column[0:4])
                print (column[0:2])
                # question_list.append(column) 
                print (column)

def list_contains(list, string):
    for item in list:
        if (item[0:4] == string):
            return True
    return False

# add
def sort_dict(list, question):
    if (question_dict.get(question)):
        if (question_dict.get(question))
    else:
        question_dict[question]
extract_questions()