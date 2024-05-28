from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import numpy as np
import pandas as pd


# 'sentence-transformers/all-MiniLM-L6-v2
def match(resume,job_des):
    print(resume,job_des)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    resume = model.encode(resume, batch_size = 10, show_progress_bar = True)
    job = model.encode(job_des, batch_size =2, show_progress_bar = True)
    similarities = cosine_similarity(resume,job)
    return similarities * 100

