import pandas as pd
import numpy as np
import torch
import re
import json
import os
import time
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from SolrHandler import solr_getRelatedDocs, solr_commit

ParagraphSimilarity_Threshold = 0.9
TextCharacter_MinLenght = 50


def load_albertModel():
    model_name = 'Albert-persiannews'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer

def split_data2Paragraphs(documents, text_field):
    paragraphs_docs = []
    paragraphs_pos = []
    for doc in documents:
        content = re.split('[.:\n]', doc[text_field])
        clean_list = [x for x in content if len(x) > TextCharacter_MinLenght and x != 'nan']
        pos = []
        for txt in clean_list:
            start = doc[text_field].index(txt),
            pos.append({'start': start[0], 'end': start[0] + len(txt) - 1})
        paragraphs_docs.append(clean_list)
        paragraphs_pos.append(pos)
    print('The related documents size: ', len(paragraphs_docs))
    return paragraphs_docs, paragraphs_pos

def get_paragraphEmbeddings(model, tokenizer, paragraphs):
    documents_embedding=[]
    for doc in paragraphs:
        doc_embedding=[]
        for para in doc:
            encoded_data = tokenizer(para, padding=True, truncation=True, return_tensors='pt')
            try:
                with torch.no_grad():
                    model_output = model(**encoded_data)
            except Exception as error:
                print(error)
                print(para)
            data_embeddings = model_output[0]
            doc_embedding.append(data_embeddings.mean(axis=1).cpu().numpy())
        documents_embedding.append(doc_embedding)
    print('Embedding extraction is done!')
    return documents_embedding

def similarity_calculation(main_docs, paragraphs_doc, paragraphs_pos, documents_embedding):
    scores = []
    for i in range(len(documents_embedding)):
        root_emb = documents_embedding[i]
        for j in range(i+1, len(documents_embedding)):
            if i+1 < len(documents_embedding):
                sum_score = []
                similar_paragraphs = []
                candidate_emb = documents_embedding[j]
                for k in range(len(root_emb)):
                    for h in range(len(candidate_emb)):
                       score = cosine_similarity(root_emb[k].mean(axis=0).reshape(1,-1),  candidate_emb[h].mean(axis=0).reshape(1,-1))
                       sum_score.append(score)
                       if score > ParagraphSimilarity_Threshold :
                            ds = {'text1': paragraphs_doc[i][k], 'text2': paragraphs_doc[j][h], 'position1': paragraphs_pos[i][k],
                                  'position2': paragraphs_pos[j][h], 'score':"{:.2f}".format(score[0][0])}
                            similar_paragraphs.append(ds)
                final_score = np.mean(sum_score)
                scores.append({'id1': main_docs[i]['id'], 'id2': main_docs[j]['id'], 'num1': main_docs[i]['num'][0] if 'num' in main_docs[i] else '',
                               'num2': main_docs[j]['num'][0] if 'num' in main_docs[j] else '', 'type1': main_docs[i]['type'], 'type2': main_docs[j]['type'],
                               'score': "{:.2f}".format(final_score), 'similar_paragraphs': similar_paragraphs})
    return scores

def run_similarDocs2JSON(clause_num):
    documents = solr_getRelatedDocs(str(clause_num))
    paragraphs_data, paragraphs_pos = split_data2Paragraphs(documents, 'text_normalized')
    embeddings = get_paragraphEmbeddings(model, tokenizer, paragraphs_data)
    similar_paragraphs = similarity_calculation(documents, paragraphs_data, paragraphs_pos, embeddings)
    with open('similarity_docs/{}.json'.format(clause_num), 'w', encoding='utf-8') as fp:
        json.dump(similar_paragraphs, fp, ensure_ascii=False)
    print('The {}.json file writed'.format(clause_num))

def save_docsInSolr(scores_dict, clause_num):
    documents = []
    for doc in scores_dict:
        for paragraph in doc['similar_paragraphs']:
            paragraph['id'] = '{}'.format(time.time())
            paragraph['clause_num'] = clause_num
            paragraph['id1'] = doc['id1']
            paragraph['id2'] = doc['id2']
            paragraph['num1'] = doc['num1']
            paragraph['num2'] = doc['num2']
            paragraph['type1'] = doc['type1']
            paragraph['type2'] = doc['type2']
            paragraph['doc_score'] = doc['score']
            paragraph['pos_start1'] = paragraph['position1']['start']
            paragraph['pos_end1'] = paragraph['position1']['end']
            paragraph['pos_start2'] = paragraph['position2']['start']
            paragraph['pos_end2'] = paragraph['position2']['end']
            del paragraph['position1']
            del paragraph['position2']
            documents.append(paragraph)
            time.sleep(.002)
    solr_commit(documents)
    return documents

def save_jsonFileInSolr(file_path, clause_num):
    file = open(file_path, 'r', encoding='utf-8')
    js = json.load(file)
    return save_docsInSolr(js, clause_num)


if __name__ == '__main__':
    model, tokenizer = load_albertModel()
    for clause_num in ['90']:
        run_similarDocs2JSON(clause_num)
    # root_path = 'similarity_docs/'
    # files = os.listdir(root_path)
    # for file in files:
    #     if file.endswith('json'):
    #         print(file)
    #         save_jsonFileInSolr(root_path + file, file.replace('.json', ''))