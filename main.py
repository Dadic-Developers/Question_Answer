import pandas as pd
import operator
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class QuestionSimilarity():
    def __init__(self):
        self.questions_path= pd.read_excel(str(Path().absolute()) + "/Question.xlsx")
        model_name = str(Path().absolute()) + '/Albert-persiannews'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.questions_path="Question.xlsx"
        self.__DataLoading()

    def __DataLoading(self):
        df_Question = pd.read_excel(self.questions_path)
        self.Questions = list(zip(df_Question.Question, df_Question.Add_Question, df_Question.Question_id,df_Question.Keyword))
        self.Answers = dict(zip(df_Question.Question_id, df_Question.Answer))
        self.__doc_embedding = []
        for question in self.Questions:

            encoded_Qu = self.tokenizer(question[0], padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = self.model(**encoded_Qu)
            Question_embeddings = model_output[0]
            self.__doc_embedding.append(Question_embeddings.mean(axis=1).cpu().numpy())

    def SimilarityCalculation(self, message):

        encoded_input = self.tokenizer(message, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        message_embeddings=model_output[0]
        message_emb= message_embeddings.mean(axis=1).cpu().numpy()
        scores=np.zeros((1,len(self.__doc_embedding)))
        for j in range(len(self.__doc_embedding)):
            scores[0, j]=cosine_similarity(message_emb.mean(axis=0).reshape(1,-1),  self.__doc_embedding[j].mean(axis=0).reshape(1,-1))

        ls_dict=[]
        for k in range(len(self.Questions)):
             if scores[0][k]>0.5 :
                print(k)
                ds = {'Question_Id': self.Questions[k][2], 'Question_Added': self.Questions[k][1].split('،'),
                         'Question_Score':scores[0][k], 'Question_Text':self.Questions[k][0]}
                ls_dict.append(ds)
        sorted_tuples = sorted(ls_dict,key=operator.itemgetter('Question_Score'),reverse=True)
        result = sorted_tuples[0:5]
        return result

    def AnswerOfQuestion(self, qid, add_answers):
        answer_template = self.Answers[qid]
        return answer_template

