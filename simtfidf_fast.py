from sklearn.feature_extraction.text import TfidfVectorizer
from string import digits
import scipy.spatial.distance
import pandas as pd
from hazm import *
import fasttext
import re
#loading fasttext model
ft = fasttext.load_model('/home/ubuntu/mehrdad/cc.fa.300.bin')

def clean(text):
    normalizer = Normalizer()
    # Remove Punctuation
    text = re.sub(r'''      # Start raw string block
               \W+       # Accept one or more non-word characters
               ''',  # Close string block
                  ' ',  # and replace it with a single space
                  text,
                  flags=re.VERBOSE)

    text = re.sub(r'[«»،]', ' ', text)
    # Remove Numbers
    remove_digits = str.maketrans('', '', digits)
    text = text.translate(remove_digits)
    # Remove English Character
    text = re.sub(r'[a-zA-Z]', '', text)
    text = re.sub(r"\s+", ' ', text)
    return text

df=pd.read_json("54.json")
stop_word_doc=pd.read_csv("stop_words.txt")['sw'].tolist()

documents=[]
data=[]
docs=[]
for index,row in df.iterrows():
    data.append(row['text'])
    doc = row['text'].replace("\n", " ")
    docs.append(clean(doc))
    content=row['text'].split('\n')
    clean_list = [x for x in content if len(x) > 50 and x != 'nan']
    if len(clean_list)<2:
        content = re.split('[.:]', clean_list[0])
        clean_list = [x for x in content if len(x) > 50]
    documents.append(clean_list)

#TfidfVectorizer Documents and feature_names
tfidf = TfidfVectorizer(min_df=0.1,max_df=0.7,stop_words=stop_word_doc)
X_tfidf = tfidf.fit_transform(docs).toarray()
vocab = tfidf.vocabulary_
reverse_vocab = {v:k for k,v in vocab.items()}
feature_names = tfidf.get_feature_names()
df_tfidf = pd.DataFrame(X_tfidf, columns = feature_names)
idx = X_tfidf.argsort(axis=1)
tfidf_max = idx[:,-10:]
df_tfidf['top'] = [[reverse_vocab.get(item) for item in row] for row in tfidf_max ]

docs_features=[]
document_vec=[]
for i in range(len(df_tfidf['top'])):
    doc_tfidf=' '.join(df_tfidf['top'][i])
    docs_features.append(doc_tfidf)
    document_vec.append(ft.get_sentence_vector(doc_tfidf))
#Vectorize paragraph of documents
doc_paragraph_vec=[]
for i in range(len(documents)):
    paragraph_vec=[]
    for j in range(len(documents[i])):
        response = tfidf.transform([documents[i][j]])
        feature_names = tfidf.get_feature_names()
        pr = []
        for col in response.nonzero()[1]:
            pr.append(feature_names[col])
        paragraph_doc = ' '.join(pr)
        paragraph_vec.append(ft.get_sentence_vector(paragraph_doc))
    doc_paragraph_vec.append(paragraph_vec)
#calculate similarity documents
similar_docs=[]
for i in range(len(data)):
    for j in range(min(i+1,len(data)), len(data)):
        score = 1-scipy.spatial.distance.cosine(document_vec[i],document_vec[j])
        if score > 0.75 :
            ds = {'id1': i, 'id2': j, 'doc1': data[i],'doc_feature1': docs_features[i],'text2': data[j],'doc_features2': docs_features[j],'score':score}
            similar_docs.append(ds)
df=pd.DataFrame(similar_docs)
df.to_excel('result_doctf75.xlsx')
#Calculate Similarity Paragraph of Documents with each other
similar_paragraphs = []
for i in range(len(documents)):
    root_doc = documents[i]
    for j in range(min(i+1,len(documents)), len(documents)):
            candidate_doc = documents[j]
            for k in range(len(root_doc)):
                for h in range(len(candidate_doc)):
                   score = 1-scipy.spatial.distance.cosine(doc_paragraph_vec[i][k],  doc_paragraph_vec[j][h])
                   if score > 0.8 :
                        ds = {'id1': i, 'id2': j,'score':score[0][0]}
                        similar_paragraphs.append(ds)

