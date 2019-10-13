import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.linear_model import LogisticRegression
import pickle
import scipy.sparse as sp
from flask import Flask, render_template
from flask import jsonify
from flask import request

app = Flask(__name__)

def striphtml(data):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(data))
    return cleantext

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

def preprocess_title(Title):
    Title=striphtml(Title.encode('utf-8'))
    Title=re.sub(r'[^A-Za-z0-9#+.\-]+',' ',Title)
    words=word_tokenize(str(Title.lower()))
    Title=' '.join(str(stemmer.stem(j)) for j in words if j not in stop_words and (len(j)!=1 or j=='c' or j=='r'))
    return Title

def preprocess_body(Body):
    is_code=0
    code_str=""
    if '<code>' in Body:
        is_code = 1
        code = re.findall('<code>(.*?)</code>', Body, flags=re.MULTILINE|re.DOTALL)
        code_str = code_str.join(code)
        
    question=re.sub('<code>(.*?)</code>', '', Body, flags=re.MULTILINE|re.DOTALL)
        
    question=striphtml(question.encode('utf-8'))
    question=re.sub(r'[^A-Za-z0-9#+.\-]+',' ',question)
    words=word_tokenize(str(question.lower()))
    question=' '.join(str(stemmer.stem(j)) for j in words if j not in stop_words and (len(j)!=1 or j=='c' or j=='r'))
    
    sent=""
    dup = dict()
    for ch in code_str:
        ch=ch.lower()
        if(ch.isalpha()):
            sent+=(ch)
        if(ch in [' ','.','(',')','{','}','[',']','_','-','/','$','&','<','>',':',';','/','\\',"'",'?','!','@','#','%','^','*','+','=','|']):
            if(ch not in dup):
                dup[ch]=(int)(1)
            if(dup.get(ch)<=5):
                dup[ch]=(int)(dup.get(ch)+1)
                sent+=" "+ch+" "
            else:
                sent+=" "

    dup = dict()
    sent1=""
    for ch in sent.split():
        if(ch not in dup):
            dup[ch]=(int)(1)
        if(dup.get(ch)<=10):
            dup[ch]=(int)(dup.get(ch)+1)
            sent1+=" "+ch+" "
        else:
            sent1+=" "


    sent1=' '.join(sent1.split())

    
    return sent1,question
	
	
def get_tags(title, code, question):
	text = title+" "+title+" "+title+" "+question+" "+code
	vector = tf1_new.transform([text])
	pred = model.predict(vector)
	
	output=[]
	response=[]
	for tag,idx in zip(tags,pred.toarray()[0]):
		if(idx==1):
			output.append(tag)
	
	return (output)



tags=""
with open('model/tags_list.txt', 'r') as f:
    tags+=(f.read())
tags = tags.split()
tags = tags[:100]
print("Tags Loaded!")

with open("model/LR_tfidf_3title_question_code_model_s.pkl",'rb') as f:
    model = pickle.load(f)
print("Model Loaded!")

tf1_vocal = pickle.load(open("model/x_tfidf_train_multilabel_vocal_s.pickle", 'rb'))
tf1_idf = pickle.load(open("model/x_tfidf_train_multilabel_idf_s.pickle", 'rb'))

tf1_new = TfidfVectorizer(min_df=0.00009, max_features=400000, tokenizer = lambda x: x.split(), ngram_range=(1,4), vocabulary=tf1_vocal)
tf1_new._tfidf._idf_diag = sp.spdiags(tf1_idf, diags = 0, m = len(tf1_idf), n = len(tf1_idf))
print("Vectorizer loaded!")



@app.route("/predict",methods=["POST"])
def predict():
	msg = request.get_json(force=True)
	title = msg['title']
	body = msg['body']
	
	title = preprocess_title(title)
	code,question = preprocess_body(body)
	
	output = get_tags(title, code, question)
	response=[]
	response.append({'name':output})
	print(response)
	
	return(jsonify(response))
	








    
    
