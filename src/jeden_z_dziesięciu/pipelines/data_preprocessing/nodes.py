"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.9
"""

import spacy
from transformers import AutoTokenizer

def filter_sentences_with_binary_answer(questions, answers):
    ret_q, ret_a = [], []
    for question, answer in zip(questions, answers):
        # yes/no questions
        if "tak" == answer or "nie" == answer:
            ret_q.append(question)
            ret_a.append(answer)
        # answer included in question
        # if answer in question:
        #    ret_q.append(question)
        #    ret_a.append(answer)
    #print(len(ret_a), len(ret_q))
    return ret_q, ret_a

def split_sentences(questions_file, answers_file):
    questions = questions_file.split("\n")[:-1]
    answers = answers_file.split("\n")[:-1]
    answers = [answer.split("\t")[0] for answer in answers]
    return questions, answers, '\n'.join([question + ' ' + answer for question, answer in zip(questions, answers)])

def make_tokenizer(model_name):
    if model_name == 'pl_core_news_sm':
        tokenizer = spacy.load(model_name)
    if model_name == "papugapt2":
        tokenizer = AutoTokenizer.from_pretrained("flax-community/papuGaPT2")
    if model_name == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if model_name == "gpt4all":
        tokenizer = AutoTokenizer.from_pretrained("nomic-ai/gpt4all-j")
    return tokenizer
