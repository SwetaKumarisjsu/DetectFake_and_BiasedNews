import pandas as pd
from csv_parser import *
"""
File mainly for IO utils 
"""

def load(csv):
    data = pd.read_csv(csv, parse_dates=[ID, TITLE, AUTHOR, TEXT, LABEL])
    return data

stop_words =['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself',
            'yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself',
            'they','them','their','theirs','themselves','what','which','who','whom','this','that',
            'these','those','am','is','are','was','were','be','been','being','have','has','had',
            'having','do','does','did','doing','a','an','the','and','but','if','or','because','as',
            'until','while','of','at','by','for','with','about','against','between','into','through',
            'during','before','after','above','below','to','from','up','down','in','out','on','off',
            'over','under','again','further','then','once','here','there','when','where','why','how',
            'all','any','both','each','few','more','most','other','some','such','no','nor','not',
            'only','own','same','so','than','too','very','s','t','can','will','just','don','should',
            'now','uses','use','using','used','one','also']

"""
Constants 
"""

ID = 'id'
TITLE = 'title'
AUTHOR = 'author'
TEXT = 'text'
LABEL = 'label'

N = 340
"""
End Constants 
"""

"""
Only for testing 
"""

if __name__ == '__main__':

    reader = read_csv(filename=file_path)

    count = 0

    for row in reader:
        print(row[TEXT])
        count += 1

        if count == 19: break

#    training = df[0:10000]
#    test = df[20000: N]


#    training.to_csv('../unclean-data/all/train_data.csv', encoding='utf-8')
#    test.to_csv("../unclean-data/all/test_data.csv", encoding='utf-8')


