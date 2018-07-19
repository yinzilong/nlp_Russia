import os
from sklearn.metrics.pairwise import pairwise_distances_argmin

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from utils import *
import datetime
from utils import RESOURCE_PATH

class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        #print("type(self.thread_embeddings_folder):", type(self.thread_embeddings_folder))
        #print("type(tag_name+.pkl)", type(tag_name +".pkl"))
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        #print("embeddings: ", embeddings_path)
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings
    
    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
            
            拿到该标签对应的候选集合
            将question转换为向量
            求最相似的问题标签
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)
        print("thread_ids:", thread_ids)
        print("thread_embeddings:", thread_embeddings)

        # HINT: you have already implemented a similar routine in the 3rd assignment.
        #### YOUR CODE HERE ####
        question_vec =  question_to_vec(question, self.word_embeddings, self.embeddings_dim)
        #### YOUR CODE HERE ####
        #从383456中找到最相似的
        #print(question_vec)
        #print(thread_embeddings)
        question_vec = question_vec.reshape(1,-1)  #只有一个样本，变成一行
        best_thread = pairwise_distances_argmin(question_vec, thread_embeddings)  
        
        print("best_thread:", best_thread[0])
        #得打印出来看一下thread_ids的组成
        return thread_ids[best_thread[0]]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])
        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(RESOURCE_PATH)

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        # Hint: you might want to create and train chatterbot.ChatBot here.
        # It could be done by creating ChatBot with the *trainer* parameter equals 
        # "chatterbot.trainers.ChatterBotCorpusTrainer"
        # and then calling *train* function with "chatterbot.corpus.english" param
        
        ########################
        #### YOUR CODE HERE ####
        ########################
        self.chatbot = ChatBot(
            "Tunan_robot",
            trainer='chatterbot.trainers.ChatterBotCorpusTrainer'
        )
        
        #使用英文语料库训练
        self.chatbot.train("chatterbot.corpus.english")
        
        
        #开始对话
        #while True:
        #    print(chatbot.get_response(input(">")))
        return self.chatbot
        
    #拿到用户的问题之后回答的过程
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""
        #　针对用户问题的意图，进行stackoverflow模块和chitchat模型的选择性调用
        
        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        """
            问题预处理
            问题向量化
            问题意图预测
        """
        prepared_question = text_prepare(question)#### YOUR CODE HERE ####
        print("after prepared_question:", datetime.datetime.now())
        features = self.tfidf_vectorizer.transform([prepared_question])#### YOUR CODE HERE ####
        print("after features:", datetime.datetime.now())
        intent = self.intent_recognizer.predict(features)#### YOUR CODE HERE ####
        print("after intent:", datetime.datetime.now())

        # Chit-chat part:   
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            response = self.create_chitchat_bot().get_response(question)#### YOUR CODE HERE ####
            print("after response:", datetime.datetime.now())
            return response
        
        # Goal-oriented part:
        else:        
            """
                问题标签分类
                拿到最好的候选匹配的id
            """
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict(features)[0]#### YOUR CODE HERE ####
            print("预测出来的tag:", tag)
            
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(prepared_question, tag)#### YOUR CODE HERE ####
            print("预测出来的thread_id:", thread_id)
            return self.ANSWER_TEMPLATE % (tag, thread_id)

