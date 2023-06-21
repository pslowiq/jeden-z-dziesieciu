"""
This is a boilerplate pipeline 'wikipedia_search'
generated using Kedro 0.18.9
"""

import requests
import re
import Levenshtein


def remove_html_tags(string):
    clean_string = re.sub(r"<.*?>", "", string)
    return clean_string


def get_wikipedia_titles(tokens):
    search_query = " ".join(tokens)
    url = f"https://pl.wikipedia.org/w/api.php?action=query&list=search&format=json&srsearch={search_query}"

    response = requests.get(url)
    data = response.json()

    titles = []
    snippets = []
    if "query" in data and "search" in data["query"]:
        for result in data["query"]["search"]:
            titles.append(result["title"])
            snippets.append(remove_html_tags("".join(result["snippet"])))

    return titles, snippets


def qualified_tokens(tokenizer, sentence):
    return [token.text for token in tokenizer(sentence) if len(token) > 1]


def get_dists(tokenizer, tokens, titles):
    candidates = []

    for title in titles:
        tokenized_title = qualified_tokens(tokenizer, title)
        # print(f'Tokens: {" ".join(tokens)} Title : {title} {Levenshtein.seqratio(" ".join(tokens), tokenized_title)}')
        is_candidate = True
        for token in tokens:
            # print('Token : ' + token)
            for title_token in tokenized_title:
                if Levenshtein.seqratio(token, title_token) > 0.5:
                    is_candidate = False
                # print(f'{token} - {title_token} : {Levenshtein.seqratio(token, title_token)}')
        if is_candidate:
            candidates.append(title)

    return candidates


def search_and_get_first_candidate(tokenizer, question):

    tokenized_question = qualified_tokens(tokenizer, question)
    titles, _ = get_wikipedia_titles(tokenized_question)
    candidates = get_dists(tokenizer, tokenized_question, titles)
    while len(candidates) == 0 and len(tokenized_question) > 0:
        tokenized_question = tokenized_question[1:]
        titles, _ = get_wikipedia_titles(tokenized_question)
        candidates = get_dists(tokenizer, tokenized_question, titles)

    if len(candidates) == 0:
        print("Couldn't find any candidate")
        return "Have no idea"

    return candidates[0]

def wikipedia_test(tokenizer, questions, answers):
    res = ""
    cnt = 0

    with open("wiki_res.txt", "w", encoding="utf-8") as f:
        for i in range(len(questions)):
            candidate = search_and_get_first_candidate(tokenizer, questions[i])
            ratio = Levenshtein.seqratio(candidate, answers[i])
            print(f"Question {i}: {questions[i]}", file=f)
            print(f"Answer {i}: {answers[i]}", file=f)
            print(f"First candidate: {candidate} | Ratio: {ratio}\n", file=f)

            res += f"Question {i}: {questions[i]}\n"
            res += f"Answer {i}: {answers[i]}\n"
            res += f"First candidate: {candidate} | Ratio: {ratio}\n\n"
            if ratio >= 0.5:
                cnt += 1

    res += f"Total correct (>= 0.5) : {cnt}, % total {cnt / len(questions)}"

    return res
