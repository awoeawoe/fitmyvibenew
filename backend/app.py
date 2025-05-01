import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
#from sqlalchemy import text
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import BertTokenizerFast, BertModel
from sentence_transformers import SentenceTransformer
import csv
import json
from pathlib import Path
# from convokit import Corpus
import faiss
import re
from safetensors.torch import load_file, save_file
from typing import List

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable

#rocchio component dump
FEEDBACK_QUERY : dict[str, List] = {}

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

app = Flask(__name__)
CORS(app)

social_embs = np.load("social-component/reddit/julia_tries_reddit_embs.npy")
print(f"Reddit embedding shape: {social_embs.shape}")

product_embs = np.load("social-component/reddit/NEW-EMBS-430.npy")
print(f"Product embedding shape: {product_embs.shape}")

#rocchio helper
def rocchio(q_vec, upvotes, downvotes, alpha=1.0, beta=0.75, gamma=0.25):
    if upvotes:
        pos_centroid = product_embs[upvotes].mean(axis=0)
    else:
        pos_centroid = 0
    if downvotes:
        neg_centroid = product_embs[downvotes].mean(axis=0)
    else:
        neg_centroid = 0

    new_query = alpha * q_vec + beta * pos_centroid - gamma * neg_centroid
    new_query /= np.linalg.norm(new_query, keepdims=True)
    return new_query.astype("float32")

#create a new entry for FEEDBACK_QUERY
def new_rocchio_record(query, query_vector, g, b, c, candidates):
    
    new_entry = {
        "q_emb" : query_vector,
        "filters" : (g, b, c),
        "candidates" : candidates, 
        "upvotes" : {},
        "downvotes" : {}
    }

    if query not in FEEDBACK_QUERY:
        FEEDBACK_QUERY[query] = []
    FEEDBACK_QUERY[query].append(new_entry)


#increment upvotes and downvotes
def acc_votes(entry: dict,
              up_dict:   dict[int | str, int],
              down_dict: dict[int | str, int]) -> None:
    for prod_id, count in up_dict.items():
        prod_id = int(prod_id)
        entry["upvotes"][prod_id] = entry["upvotes"].get(prod_id, 0) + int(count)

    for prod_id, count in down_dict.items():
        prod_id = int(prod_id)
        entry["downvotes"][prod_id] = entry["downvotes"].get(prod_id, 0) + int(count)


@app.route("/")
def home():
    return render_template('base.html',title="sample html")

def cosine_similarity(a: np.ndarray, b: np.ndarray):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)

def vectorize_query(query):
    """
    Vectorizes the ad-hoc query using pre-trained BERT embeddings.
    """

    model = SentenceTransformer('fashion-bert-output-v4')
    encoded_query = model.encode([query], convert_to_numpy=True) #tokenizer(query, return_tensors='pt', padding=True, truncation=True)
    encoded_query = encoded_query / np.linalg.norm(encoded_query, axis=1, keepdims=True)

    encoded_query = encoded_query.astype("float32")
    print(f"Query embedding shape: {encoded_query.shape}")
    return encoded_query # query_embeddings

def vector_from_id(article_id):
    """
    Returns the associated embedding vector to a certain article ID.
    Article IDs are specified in the database, where each article is associated with
    an ID as its primary key.
    Format of the return value is a tuple, where the first value is the product ID
    and the second is the embedding represented as a list of decimals.
    """
    return product_embs[article_id]

def order_articles(query_embeddings, filtered_ids, article_vectors):
    """
    Orders the articles in the database based on a cosine similarity metric.
    Takes the embedding of the query and the vectors of the searched articles as
    input, and returns a ranked list of article IDs based on similarity scores.
    """
    k_corpus = 10
    k_prod = min(len(filtered_ids), 20)
    alpha, beta = 1.0, 0.75

    q_emb = query_embeddings

    sim_u = cosine_similarity(q_emb, social_embs)  # (1, N)
    idxs_u = np.argsort(sim_u[0])[::-1][:k_corpus]
    top_u_embs = social_embs[idxs_u]
    
    expanded = alpha * q_emb + beta * top_u_embs.mean(axis=0, keepdims=True)
    expanded /= np.linalg.norm(expanded, axis=1, keepdims=True)

    def build_search_dict(filtered_ids):
        count = 0
        search_dict = {}
        for id in filtered_ids:
            search_dict[count] = id
            count += 1
        return search_dict
    
    search_dict = build_search_dict(filtered_ids)

    sim_p = cosine_similarity(expanded, article_vectors)  # (1, N)
    idxs_p = np.argsort(sim_p[0])[::-1][:k_prod]

    print("IDXS_P PRINTING")
    print(idxs_p)
    return [search_dict[search_id] for search_id in idxs_p]


def table_lookup(indices):
    """
    Looks up the relevant data about a set of articles given their article IDs.
    In its current form, this lookup returns information about the product name,
    its regular price, and a link to the image.
    """
    
    items_path = Path("COMBINED-FINAL-DEDUPED-CLEAN2.json")
    with items_path.open("r", encoding="utf-8") as f:
        items_data = json.load(f)

    items_by_id = {item["ID"]: item for item in items_data}

    def truncate_description(description, word_limit=20):
        """Truncates description to specified word limit and adds '...' if truncated"""
        if not description:
            return ""
        words = description.split()
        if len(words) <= word_limit:
            return description
        return " ".join(words[:word_limit]) + "..."

    ranked_results = []
    for idx in indices:
        rec = items_by_id.get(idx)
        if (rec):
            img_link = rec.get("prodImgLink") or "static/images/clothing-icon.png"
            prod_link = rec.get("prodLink", "") or "https://www.mercari.com/jp/"
            ranked_results.append({
                "prodName": rec.get("name"),
                "prodPrice": rec.get("price"),
                "prodImgLink": img_link,
                "prodLink": prod_link,
                "prodDesc": truncate_description(rec.get("description"))
            })
    return ranked_results

@app.route("/articles")
def episodes_search():
    query = request.args.get("inspirationDesc")

    gender = request.args.get("gender", default=None)
    if gender == "men":
        gender = "m"
    elif gender == "women":
        gender = "f"
    else:
        gender = None

    budget = request.args.get("budget", default=None)
    if budget == "":
        budget == None
    else:
        budget = float(budget)

    article = request.args.get("article", default=None)
    if article == "T":
        article = "Tops"
    elif article == "B":
        article = "Bottoms"
    elif article == "S":
        article = "Shoes"
    elif article == "A":
        article = "Accessories"
    else:
        article = None

    query_vector = vectorize_query(query)

    items_path = Path("COMBINED-FINAL-DEDUPED-CLEAN2.json")
    with items_path.open("r", encoding="utf-8") as f:
        items_data = json.load(f)

    items_by_id = {item["ID"]: item for item in items_data}

    
    def check_filters():
        filter_ids = []
        for id in range(0, 1332):
            rec = items_by_id.get(id)
            if rec == None:
                continue

            gender_filter = True if gender == "" or gender == None else rec['gender'] == gender
            if not(isinstance(budget, float)) or (rec["price"] == ""):
                budget_filter = True
            else:
                budget_low = budget - 24
                budget_high = budget + 25
                budget_filter = float(rec['price']) >= budget_low and float(rec['price']) <= budget_high
            article_filter = True if article == "" or article == None else rec['category'] == article

            # print(f"Filters for article {id}: G={gender_filter}, B={budget_filter}, A={article_filter}")

            if (gender_filter and budget_filter and article_filter):
                filter_ids.append(id)
                # print(f"ADDED IDX {id} TO CANDIDATES")
        
        return filter_ids
    
    filter_ids = check_filters()
    article_vectors = product_embs[filter_ids]

    # Articles that pass the filter are stored in article_vectors
    # Make order articles use the article vectors as the set of articles to query

    # is the bottom still relevant????

    if query not in FEEDBACK_QUERY:
        new_rocchio_record(query, query_vector, gender, budget, article, filter_ids)
        rec = FEEDBACK_QUERY[query][-1]
    else:
        rec = None
        for r in FEEDBACK_QUERY[query]:
            if r["filters"] == (gender, budget, article):
                rec = r
                break
        if rec is None:
            new_rocchio_record(query, query_vector, gender, budget, article, filter_ids)
            rec = FEEDBACK_QUERY[query][-1]

    ranked_idx   = order_articles(query_vector, filter_ids, article_vectors)
    ranked_results = table_lookup(ranked_idx)

    print("DONE RANKING")
    #print(ranked_results)
    return json.dumps(ranked_results, default=str)

@app.route("/feedback")
def feedback():
    data = request.get_json(force=True)
    query = data["query"]
    query_emb = data["q_emb"]
    filters = data["filters"]
    up = data.get("upvotes", {})
    down = data.get("downvotes", {})

    if query not in FEEDBACK_QUERY:
        return json.dumps({"error": "unknown query"}), 400

    entry_list = FEEDBACK_QUERY[query]

    ranked_results = []
    for entry in entry_list:
        if entry["filters"] == filters:
            acc_votes(entry, up, down)
            new_query = rocchio(query_emb, up, down)
            entry["q_emb"] = new_query

            candidates = entry["candidates"]
            ranked_idx = order_articles(new_query, candidates, product_embs[candidates])
            ranked_results = table_lookup(ranked_idx)
            break
    
    return json.dumps(ranked_results, default=str)


