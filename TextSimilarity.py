import pandas as pd
import numpy as np
import torch
import re
import json
import pickle
import time
import string
from string import digits
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
# import fasttext
from SolrHandler import solr_getRelatedDocs, solr_commit, solr_getAllTextDocs

class DocumentsSimilarity():

    def __init__(self, vectorization_method='tfidf', para_sim_threshold=0.8, min_txt_len=50,
                        max_doc_word_count=30, max_para_word_count=10, min_para_word=4):
        self.VectorizationMethod = vectorization_method
        self.__ParagraphSimilarity_Threshold = para_sim_threshold
        self.__TextCharacter_MinLenght = min_txt_len
        self.__MaxWordsCount_InDoc = max_doc_word_count
        self.__MaxWordsCount_InParagraph = max_para_word_count
        self.__MinTopWords_InParagraph = min_para_word
        if self.VectorizationMethod == 'tfidf':
            self.__load_tfIdfVectorize()
            print('The TF-IDF model is created')
            # self.__fasttext = fasttext.load_model('tfidf_data/cc.fa.300.bin')
            self.__fasttext = None
            print('The fasttext model is loaded')
        elif self.VectorizationMethod == 'albert':
            self.__load_albertModel()

    def __load_albertModel(self):
        model_name = 'Albert-persiannews'
        self.__albert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.__albert_model = AutoModel.from_pretrained(model_name)

    def __load_tfIdfVectorize(self):
        stop_words = pd.read_csv("stop_words.txt")['sw'].tolist()
        self.__tfidf = TfidfVectorizer(min_df=0.01, max_df=0.8, stop_words=stop_words, preprocessor=self.__clean_textData)
        corpus = pickle.load(open('tfidf_data/tfidf_corpus.pkl', 'rb'))
        self.__tfidf.fit_transform(corpus).toarray()
        self.__all_words = self.__tfidf.get_feature_names()

    def __create_rulesCorpus(self):
        docs = solr_getAllTextDocs()
        corpus = []
        for doc in docs:
            text = ''
            if 'subject_normalized' in doc:
                text = doc['subject_normalized']
            text += '\n\n' + doc['text_normalized']
            corpus.append(text)
        file = open('tfidf_data/tfidf_corpus.pkl', 'wb')
        pickle.dump(corpus, file)
        file.close()

    def __clean_textData(self, text):
        if text:
            text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
            text = re.sub(r'[«»،؛؟]', ' ', text)
            remove_digits = str.maketrans('', '', digits)
            text = text.translate(remove_digits)
            text = re.sub(r'[a-zA-Z۰-۹]', '', text)
            text = re.sub(r"\s+", ' ', text)
        return text

    def split_data2Paragraphs(self, documents, text_field):
        paragraphs_docs = []
        paragraphs_pos = []
        for doc in documents:
            content = re.split('[.:\n]', doc[text_field])
            clean_list = [x for x in content if len(x) > self.__TextCharacter_MinLenght and x != 'nan']
            pos = []
            for txt in clean_list:
                start = doc[text_field].index(txt),
                pos.append({'start': start[0], 'end': start[0] + len(txt) - 1})
            paragraphs_docs.append(clean_list)
            paragraphs_pos.append(pos)
        print('The related documents size: ', len(paragraphs_docs))
        return paragraphs_docs, paragraphs_pos

    def get_paragraphEmbeddings(self, docs_paragraphs):
        documents_embedding = []
        for doc in docs_paragraphs:
            paragraphs_embedding = []
            for paragraph in doc:
                encoded_data = self.__albert_tokenizer(paragraph, padding=True, truncation=True, return_tensors='pt')
                try:
                    with torch.no_grad():
                        model_output = self.__albert_model(**encoded_data)
                except Exception as error:
                    print(error)
                    print(paragraph)
                paragraphs_embedding.append(model_output[0].mean(axis=1).cpu().numpy())
            documents_embedding.append((paragraphs_embedding,))
        print('Embedding extraction is done!')
        return documents_embedding

    def __get_topTFIDFWords(self, text, max_words):
        vec_tfidf = self.__tfidf.transform(text)
        vec_tfidf.data.sort()
        size = len(vec_tfidf.data)
        start = size - max_words if size > max_words else 0
        indexes = vec_tfidf.indices[start:size]
        top_words = [self.__all_words[i] for i in indexes]
        return top_words

    def get_paragraphTFIDf_FastTextVectors(self, docs_paragraphs):
        documents_embedding=[]
        for doc in docs_paragraphs:
            doc_tfidf = self.__get_topTFIDFWords(['\n'.join(doc)], self.__MaxWordsCount_InDoc)
            doc_embedding = self.__fasttext.get_sentence_vector(' '.join(doc_tfidf))
            paragraphs_embedding = []
            for paragraph in doc:
                para_tfidf = self.__get_topTFIDFWords([paragraph], self.__MaxWordsCount_InParagraph)
                if len(para_tfidf) > self.__MinTopWords_InParagraph:
                    ft_embedding = self.__fasttext.get_sentence_vector(' '.join(para_tfidf))
                else:
                    ft_embedding = None
                paragraphs_embedding.append(ft_embedding)
            documents_embedding.append((paragraphs_embedding, doc_embedding))
        print('Embedding extraction is done!')
        return documents_embedding

    def similarity_calculation(self, main_docs, paragraphs_doc, paragraphs_pos, documents_embedding):
        scores = []
        for i in range(len(documents_embedding)):
            root_emb = documents_embedding[i][0]
            for j in range(i+1, len(documents_embedding)):
                if i+1 < len(documents_embedding):
                    sum_score = []
                    similar_paragraphs = []
                    candidate_emb = documents_embedding[j][0]
                    for k in range(len(root_emb)):
                        for h in range(len(candidate_emb)):
                            if self.VectorizationMethod == 'albert':
                                score = cosine_similarity(root_emb[k].mean(axis=0).reshape(1,-1),  candidate_emb[h].mean(axis=0).reshape(1,-1))
                                sum_score.append(score)
                            elif self.VectorizationMethod == 'tfidf':
                                score = 0.0
                                if root_emb[k] and candidate_emb[h]:
                                    score = cosine_similarity(root_emb[k].reshape(1, -1), candidate_emb[h].reshape(1, -1))
                            if score > self.__ParagraphSimilarity_Threshold:
                                ds = {'text1': paragraphs_doc[i][k], 'text2': paragraphs_doc[j][h], 'position1': paragraphs_pos[i][k],
                                      'position2': paragraphs_pos[j][h], 'score':"{:.2f}".format(score[0][0])}
                                similar_paragraphs.append(ds)
                    if self.VectorizationMethod == 'albert':
                        final_score = np.mean(sum_score)
                    elif self.VectorizationMethod == 'tfidf':
                        final_score = cosine_similarity(documents_embedding[i][1].reshape(1, -1),  documents_embedding[j][1].reshape(1, -1))[0][0]
                    scores.append({'id1': main_docs[i]['id'], 'id2': main_docs[j]['id'], 'num1': main_docs[i]['num'][0] if 'num' in main_docs[i] else '',
                                   'num2': main_docs[j]['num'][0] if 'num' in main_docs[j] else '', 'type1': main_docs[i]['type'], 'type2': main_docs[j]['type'],
                                   'score': "{:.2f}".format(final_score), 'similar_paragraphs': similar_paragraphs})
            print('SimilarityCalc - RootDoc: {}'.format(i))
        return scores

    def run_similarDocs2JSON(self, clause_num):
        documents = solr_getRelatedDocs(str(clause_num))
        paragraphs_data, paragraphs_pos = self.split_data2Paragraphs(documents, 'text_normalized')
        if self.VectorizationMethod == 'tfidf':
            embeddings = self.get_paragraphTFIDf_FastTextVectors(paragraphs_data)
        elif self.VectorizationMethod == 'albert':
            embeddings = self.get_paragraphEmbeddings(paragraphs_data)
        similar_paragraphs = self.similarity_calculation(documents, paragraphs_data, paragraphs_pos, embeddings)
        with open('similarity_docs/{}.json'.format(clause_num), 'w', encoding='utf-8') as fp:
            json.dump(similar_paragraphs, fp, ensure_ascii=False)
        print('The {}.json file writed'.format(clause_num))

    def save_docsInSolr(self, scores_dict, clause_num):
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

    def save_jsonFileInSolr(self, file_path, clause_num):
        file = open(file_path, 'r', encoding='utf-8')
        js = json.load(file)
        return self.save_docsInSolr(js, clause_num)


if __name__ == '__main__':
    ds = DocumentsSimilarity(vectorization_method='albert', para_sim_threshold=0.9,)
    for clause_num in range(2,283):
        ds.run_similarDocs2JSON(str(clause_num))
    # root_path = 'similarity_docs/'
    # import os
    # files = os.listdir(root_path)
    # for file in files:
    #     if file.endswith('json'):
            # print(file)
            # ds.save_jsonFileInSolr(root_path + file, file.replace('.json', ''))