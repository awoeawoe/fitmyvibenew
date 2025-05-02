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
QUERY_STATES : dict[int, dict[int, (int, int)]] = {}
NUM_QUERIES = 0

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

app = Flask(__name__)
CORS(app)

# print(f"Reddit embedding shape: {social_embs.shape}")
# del social_embs_1
# del social_embs_2

# WORKING TEST WITH RE ALIGNED EMBEDDINGS + COMMENTS JSON
# social_embs = np.load("social-component/reddit/julia_tries_reddit_embs.npy", allow_pickle=True)

social_embs1 = np.load("social-component/reddit/complete_reddit_embs1.npy", allow_pickle=True)
social_embs2 = np.load("social-component/reddit/complete_reddit_embs2.npy", allow_pickle=True)
social_embs3 = np.load("social-component/reddit/complete_reddit_embs3.npy", allow_pickle=True)

social_embs = np.concatenate((social_embs1, social_embs2), axis=0)
social_embs = np.concatenate((social_embs, social_embs3), axis=0)
print(f"Social embedding shape: {social_embs.shape}")

product_embs = np.load("social-component/reddit/prod_embs_better.npy", allow_pickle=True)
print(f"Product embedding shape: {product_embs.shape}")

comments_path = Path("social-component/reddit/filtered_texts_reddit.json")
with comments_path.open("r", encoding="utf-8") as f:
    soc_info = json.load(f)
print(f"Social info shape: {len(soc_info)}")

soc_count = 0
socinfo_id_to_text = {}
for entry in soc_info:
    socinfo_id_to_text[soc_count] = entry
    soc_count += 1

merge_state_dict = {}
files = ["tensor_pack/chunk_1_1.safetensors",
         "tensor_pack/chunk_1_2.safetensors",
         "tensor_pack/chunk_1_3.safetensors",
         "tensor_pack/chunk_1_4.safetensors",
         "tensor_pack/chunk_2.safetensors",
         "tensor_pack/chunk_3.safetensors",
         "tensor_pack/chunk_4.safetensors",
         "tensor_pack/chunk_5.safetensors"]

merged_file = "fashion-bert-output-v4/model.safetensors"

def merge_files(files):
    for file in files:
        load_files_dict = load_file(file)
        merge_state_dict.update(load_files_dict)
    
merge_files(files)

save_file(merge_state_dict, merged_file)
del merge_state_dict

#rocchio helper - Updated to handle empty lists better
def rocchio(q_vec, upvotes, downvotes, alpha=1.0, beta=0.75, gamma=0.25):
    # Handle empty lists
    # if upvotes and len(upvotes) > 0:
    #     try:
    #         pos_centroid = product_embs[upvotes].mean(axis=0)
    #     except Exception as e:
    #         print(f"Error calculating positive centroid: {e}")
    #         pos_centroid = np.zeros_like(q_vec)
    # else:
    #     pos_centroid = np.zeros_like(q_vec)
    
    # if downvotes and len(downvotes) > 0:
    #     try:
    #         neg_centroid = product_embs[downvotes].mean(axis=0)
    #     except Exception as e:
    #         print(f"Error calculating negative centroid: {e}")
    #         neg_centroid = np.zeros_like(q_vec)
    # else:
    #     neg_centroid = np.zeros_like(q_vec)
    pos_vec_list = []
    for idx in upvotes:
        pos_vec_list.append(product_embs[idx])
    if (len(pos_vec_list) == 0):
        pos_centroid = np.zeros_like(q_vec)
    else:
        pos_vec_list = np.array(pos_vec_list)
        pos_centroid = pos_vec_list.mean(axis=0)

    neg_vec_list = []
    for idx in downvotes:
        neg_vec_list.append(product_embs[idx])
    if (len(neg_vec_list) == 0):
        neg_centroid = np.zeros_like(q_vec)
    else:
        neg_vec_list = np.array(neg_vec_list)
        neg_centroid = neg_vec_list.mean(axis=0)

    new_query = alpha * q_vec + beta * pos_centroid - gamma * neg_centroid
    
    # Check for zero vector
    norm = np.linalg.norm(new_query)
    if norm > 0:
        new_query = new_query / norm
    
    return new_query.astype("float32")

#create a new entry for FEEDBACK_QUERY
def new_rocchio_record(query, query_vector, g, b, c, candidates):
    
    new_entry = {
        "q_id" : NUM_QUERIES,
        "q_emb" : query_vector,
        "filters" : (g, b, c),
        "candidates" : candidates
    }

    if query not in FEEDBACK_QUERY:
        FEEDBACK_QUERY[query] = []
    FEEDBACK_QUERY[query].append(new_entry)


#increment upvotes and downvotes
# def acc_votes(entry: dict,
#               up_dict:   dict[int | str, int],
#               down_dict: dict[int | str, int]) -> None:
#     for prod_id, count in up_dict.items():
#         prod_id = int(prod_id)
#         entry["upvotes"][prod_id] = entry["upvotes"].get(prod_id, 0) + int(count)

#     for prod_id, count in down_dict.items():
#         prod_id = int(prod_id)
#         entry["downvotes"][prod_id] = entry["downvotes"].get(prod_id, 0) + int(count)


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

def get_relevant_comments(query_embeds):
    k_corpus = 10
    # q_emb = query_embeds

    sim_u   = cosine_similarity(query_embeds, social_embs)
    idxs_u  = np.argsort(sim_u[0])[::-1][:k_corpus]
    idxs_py = idxs_u.tolist()

    return [socinfo_id_to_text[i] for i in idxs_py if i in socinfo_id_to_text]

def order_articles(query_embeddings, filtered_ids, article_vectors):
    """
    Orders the articles in the database based on a cosine similarity metric.
    Takes the embedding of the query and the vectors of the searched articles as
    input, and returns a ranked list of article IDs based on similarity scores.
    """
    k_corpus = 30
    k_prod = min(len(filtered_ids), 25)
    alpha, beta = 0.9, 1.3

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
    # print(search_dict)

    sim_p = cosine_similarity(expanded, article_vectors)  # (1, N)
    sim_top = [np.round(score, 3) for score in np.sort(sim_p[0])[::-1][:k_prod]]
    idxs_p = np.argsort(sim_p[0])[::-1][:k_prod]

    searched_articles = [search_dict[search_id] for search_id in idxs_p]
    articles_and_scores = list(zip(searched_articles, sim_top))
    # print(articles_and_scores)

    print("IDXS_P PRINTING")
    print(idxs_p)
    return articles_and_scores

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

    def truncate_description(description, word_limit=17):
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
                "prodDesc": truncate_description(rec.get("description")),
                "prodId": idx,  # Include the product ID
                "prodUpvotes": QUERY_STATES[NUM_QUERIES][idx][0],
                "prodDownvotes": QUERY_STATES[NUM_QUERIES][idx][1]
            })
    return ranked_results

@app.route("/articles")
def episodes_search():

    global NUM_QUERIES
    global QUERY_STATES
    global FEEDBACK_QUERY
    NUM_QUERIES += 1

    query = request.args.get("inspirationDesc")
    gender = request.args.get("gender", default=None)
    budget = request.args.get("budget", default=None)
    if budget != "":
        budget_raw = int(budget)
        mod3 = budget_raw % 3 + 1
        div3 = budget_raw / 3 + 1

        base = 10 ** div3

        budget = base * mod3

    article = request.args.get("article", default=None)

    ##### CASE 1 - OLD QUERY #####
    if query in FEEDBACK_QUERY:
        for logged in FEEDBACK_QUERY[query]:
            if logged['filters'] == (gender, budget, article):
                print("ENTERING CASE 1: OLD QUERY")
                query_vector = logged['q_emb']

                old_query_id = logged['q_id']

                # Remove old query from log
                FEEDBACK_QUERY[query] = [new_q for new_q in FEEDBACK_QUERY[query] if new_q['filters'] != (gender, budget, article)]
                new_rocchio_record(query, query_vector, gender, budget, article, logged['candidates'])

                # Copy old set of upvotes and downvotes for this query
                QUERY_STATES[NUM_QUERIES] = QUERY_STATES[old_query_id]

                # Recalculating embedding
                upvote_ids = []
                for idx in QUERY_STATES[NUM_QUERIES].keys():
                    upvotes, _ = QUERY_STATES[NUM_QUERIES][idx]
                    for _ in range(upvotes):
                        upvote_ids.append(idx)

                downvote_ids = []
                for idx in QUERY_STATES[NUM_QUERIES].keys():
                    _, downvotes = QUERY_STATES[NUM_QUERIES][idx]
                    for _ in range(downvotes):
                        downvote_ids.append(idx)
                
                updated_query = rocchio(query_vector, upvote_ids, downvote_ids)
                
                # Lookup
                ranked_ids_and_scores = order_articles(updated_query, logged['candidates'], product_embs[logged['candidates']])
                ranked_idx = [idx for idx, _ in ranked_ids_and_scores]
                ranked_scores = [score for _, score in ranked_ids_and_scores]
                ranked_results = table_lookup(ranked_idx)

                # Attaching sim scores
                count = 0
                for result in ranked_results:
                    result['simScore'] = ranked_scores[count]
                    count += 1

                print("DONE RANKING")
                return json.dumps(ranked_results, default=str)

    ##### CASE 2 - NEW QUERY #####
    query_vector = vectorize_query(query)

    # Create new query result state for possible reverting
    QUERY_STATES[NUM_QUERIES] = {}
    for prod_id in range(len(product_embs)):
        QUERY_STATES[NUM_QUERIES][prod_id] = (0, 0)

    items_path = Path("COMBINED-FINAL-DEDUPED-CLEAN3.json")
    with items_path.open("r", encoding="utf-8") as f:
        items_data = json.load(f)

    items_by_id = {item["ID"]: item for item in items_data}

    def check_filters():
        filter_ids = []
        for id in range(len(product_embs)):
            rec = items_by_id.get(id)
            if rec == None:
                continue

            gender_filter = True if gender == "" or gender == None else rec['gender'] == gender

            if not(isinstance(budget, float)) or (rec["price"] == ""):
                budget_filter = True
            else:
                budget_filter = float(rec['price']) <= budget
            article_filter = True if article == "" or article == None else rec['category'] == article

            # print(f"Filters for article {id}: G={gender_filter}, B={budget_filter}, A={article_filter}")

            if (gender_filter and budget_filter and article_filter):
                filter_ids.append(id)
        
        return filter_ids
    
    filter_ids = check_filters()
    # Create record for Rocchio's
    new_rocchio_record(query, query_vector, gender, budget, article, filter_ids)
    article_vectors = product_embs[filter_ids]

    # Articles that pass the filter are stored in article_vectors
    # Make order articles use the article vectors as the set of articles to query

    ranked_ids_and_scores = order_articles(query_vector, filter_ids, article_vectors)
    ranked_idx = [idx for idx, _ in ranked_ids_and_scores]
    ranked_scores = [score for _, score in ranked_ids_and_scores]
    ranked_results = table_lookup(ranked_idx)

    # Attaching sim scores
    count = 0
    for result in ranked_results:
        result['simScore'] = ranked_scores[count]
        count += 1

    print("DONE RANKING")
    return json.dumps(ranked_results, default=str)  

@app.route("/comments", methods=["POST"])
def comments():
    data = request.get_json(force=True)
    query = data.get("query", "")
    query_vec = vectorize_query(query)
    comments = get_relevant_comments(query_vec)
    return comments

@app.route("/save_vote", methods=['POST'])
def save_vote():

    global QUERY_STATES

    data = request.get_json()
    product_id = data.get("product_id")
    vote_type = data.get("vote_type")
    vote_value = data.get("vote_value", 1)  # default to 1 if not spec
    query = data.get("query")
    gender = data.get("gender")

    budget = data.get("budget")
    budget_raw = int(budget)
    mod3 = budget_raw % 3 + 1
    div3 = budget_raw / 3 + 1
    base = 10 ** div3
    budget = base * mod3
    

    article = data.get("article")
    
    print(f"Vote received: {vote_type} (value: {vote_value}) for product {product_id} on query: {query}")
    print(f"(Current filter settings: {gender}; {budget}; {article})")
    
    if not query or not product_id or vote_type not in ['up', 'down']:
        return json.dumps({"error": "Invalid data"}), 400
    
    try:
        product_id = int(product_id)
    except ValueError:
        return json.dumps({"error": "Invalid product ID"}), 400
    
    if query not in FEEDBACK_QUERY:
        return json.dumps({"error": "Unknown query"}), 400
    
    # Logging vote
    former_tuple = QUERY_STATES[NUM_QUERIES][product_id]

    if vote_type == 'up':
        if vote_value == 1:
            QUERY_STATES[NUM_QUERIES][product_id] = (former_tuple[0] + 1, former_tuple[1])
        else:
            QUERY_STATES[NUM_QUERIES][product_id] = (former_tuple[0] - 1, former_tuple[1])
    else: #this is for downvotes
        if vote_value == 1:
            QUERY_STATES[NUM_QUERIES][product_id] = (former_tuple[0], former_tuple[1] + 1)
        else: #if the value is 0
            QUERY_STATES[NUM_QUERIES][product_id] = (former_tuple[0], former_tuple[1] - 1)

    return json.dumps({"success": True, "message": f"{vote_type} vote recorded for product {product_id}"}), 200


# @app.route("/feedback", methods=['GET', 'POST'])
# def feedback():
#     try:
#         data = request.get_json(force=True)
#         print(data)
#         query = data["query"]
#         filters_list = data["filters"]  
#         gender, budget, article = filters_list
        
#         upvotes = data.get("upvotes", {})
#         downvotes = data.get("downvotes", {})

#         if query not in FEEDBACK_QUERY:
#             return json.dumps({"error": "unknown query"}), 400

#         entry_list = FEEDBACK_QUERY[query]

#         ranked_results = []
#         filters_tuple = (gender, budget, article)
        
#         for entry in entry_list:
#             if entry["filters"] == filters_tuple:
#                 upvotes_int = {int(k): v for k, v in upvotes.items()}
#                 downvotes_int = {int(k): v for k, v in downvotes.items()}
                
#                 upvote_ids = list(upvotes_int.keys())
#                 downvote_ids = list(downvotes_int.keys())
                
#                 print(f"Applying Rocchio with upvotes: {upvote_ids}, downvotes: {downvote_ids}")
                
#                 # apply Rocchio algorithm - only if we have votes
#                 if upvote_ids or downvote_ids:
#                     new_query = rocchio(
#                         entry["q_emb"], 
#                         upvote_ids,
#                         downvote_ids
#                     )
                    
#                     # update the entry with the new query
#                     entry["q_emb"] = new_query
#                 else:
#                     new_query = entry["q_emb"]

#                 # candidate product IDs
#                 candidates = entry["candidates"]
                
#                 if not candidates:
#                     return json.dumps({"error": "No candidate products found"}), 400
                
#                 # get article vectors for these candidates
#                 article_vectors = product_embs[candidates]
            
#                 # oorder articles based on the new query
#                 ranked_idx = order_articles(new_query.reshape(1, -1), candidates, article_vectors)
                
#                 # look up product details
#                 ranked_results = table_lookup(ranked_idx)
#                 break
        
#         if not ranked_results:
#             return json.dumps({"error": "No matching entry found or no results after reranking"}), 400
        
#         return json.dumps(ranked_results, default=str)
    
#     except Exception as e:
#         import traceback
#         print(f"Error in feedback route: {str(e)}")
#         print(traceback.format_exc())
#         return json.dumps({"error": f"Server error: {str(e)}"}), 500