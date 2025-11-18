import pickle, numpy as np
from embed import get_emb_from_path

def enroll_many(db_path, person_id, img_paths):
    db = load_db(db_path)
    if not db:
        return None
    embs = []
    for p in img_paths:
        e = get_emb_from_path(p)
        if e is not None: embs.append(e)
    if not embs: return False
    centroid = np.mean(np.stack(embs,0), axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    db[person_id] = centroid
    with open(db_path,'wb') as f: pickle.dump(db,f)
    return True

def identify(db_path, query_path, threshold=0.6):
    db = load_db(db_path)
    q = get_emb_from_path(query_path)
    if q is None: return None
    best_id, best_score = None, -1.0
    for pid, cent in db.items():
        score = float(np.dot(q, cent))
        if score > best_score:
            best_score, best_id = score, pid
    if best_score >= threshold: return best_id, best_score
    return None

def load_db(path='face_db.pkl'):
    try:
        return pickle.load(open(path,'rb'))
    except:
        return {}