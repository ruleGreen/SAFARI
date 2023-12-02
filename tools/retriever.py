# coding: utf-8
import rocketqa
import sys; sys.path.append('./')
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from tools.retrieval.encoder_corpus import Encoder

def cosine_similarity(vector_a, vector_b):
    return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))

# support BM25, DPR, RocketQAv2, ....
class Retriever:
    def __init__(self, model_name="") -> None:
        self.model_name = model_name
        if model_name == "bm25":
            model_path = ""
        elif model_name == "dpr":
            model_path = "/kfdata03/kf_grp/hrwang/retriever/universal_retriever/baselines/Multi-CPR/retrieval/output_model/kbp_model/bert-base-chinese_cls/"
        elif model_name == "rocketqa":
            model_path = "/kfdata03/kf_grp/hrwang/retriever/universal_retriever/baselines/RocketQA/kbp_models/"  # Your trained rocketqav2 model
        else:
            model_path = ""
        self.model_path = model_path
        self.init_model()

    def init_model(self):
        if self.model_name == "rocketqa":
            self.retriever = rocketqa.load_model(model=self.model_path + "config.json", use_cuda=True, device_id=0)
        elif self.model_name == "dpr":
            self.retriever = Encoder(self.model_path, "cls", 512)  # pool type as cls and max sequence length as 512

    def retrieve_top_n(self, context, database, number=1, return_score=False):
        # preprocess context to be consistent with the retriever
        context = context.replace("\n", " ").replace("用户: ", "").replace("系统: ", "")
        if self.model_name == "rocketqa":
            query_list = [context] * len(database)
            relevance = self.retriever.matching(query=query_list, para=database)
            sim_scores = list(relevance)
            sorted_scores = sorted(sim_scores, reverse=True)[:number]
            top_n_score_index = [sim_scores.index(score) for score in sorted_scores]
            return [database[index] for index in top_n_score_index]
        elif self.model_name == "bm25":
            tokenized_corpus = [list(jieba.cut(doc)) for doc in database]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = list(jieba.cut(context))
            doc_scores = bm25.get_scores(tokenized_query)
            return bm25.get_top_n(tokenized_query, database, n=number)
        elif self.model_name == "dpr":
            qvec = self.retriever.predict([context])
            candidate_vecs = self.retriever.predict(database)
            sim_scores = [cosine_similarity(qvec, candidate_vecs[i]) for i in range(len(candidate_vecs))]
            sorted_scores = sorted(sim_scores, reverse=True)[:number]
            top_n_score_index = [sim_scores.index(score) for score in sorted_scores]
            # if return_score:
            #     return [[database[index], top_n_score_index[index]] for index in top_n_score_index]
            return [database[index] for index in top_n_score_index]

if __name__ == "__main__":
    retriever = Retriever("bm25")
    retriever.retrieve_top_n(["你叫啥名字？"], ["我叫王献之", "我的家在东北"])