import streamlit as st
from PIL import Image
import numpy as np
import pickle
import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


st.set_page_config(page_title="Face Recognition", layout="centered")

mtcnn = MTCNN(image_size=160, margin=14, keep_all=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to()

# Загрузка порога (EER) если есть
DEFAULT_THRESHOLD = 0.53837847709
eer_threshold = DEFAULT_THRESHOLD
if os.path.exists("scores.npz"):
    try:
        tmp = np.load("scores.npz", allow_pickle=True)
        if "eer_threshold" in tmp:
            eer_threshold = float(tmp["eer_threshold"])
    except Exception as e:
        st.warning(f"Не удалось загрузить scores.npz: {e}")

# Загрузка базы лиц: предпочитаем face_db.pkl (centroids)
# иначе X.npy/y.npy
def load_face_db(path="face_db.pkl"):
    if os.path.exists(path):
        try:
            with open(path,'rb') as f:
                db = pickle.load(f)
            return db
        except Exception as e:
            st.warning(f"Не удалось загрузить face_db.pkl: {e}")
    return {}

def load_embeddings_db(x_path="X.npy", y_path="y.npy"):
    if os.path.exists(x_path) and os.path.exists(y_path):
        try:
            X = np.load(x_path)
            y = np.load(y_path)
            return X, y
        except Exception as e:
            st.warning(f"Не удалось загрузить X.npy/y.npy: {e}")
    return None, None

face_db = load_face_db()
X_emb, y_labels = load_embeddings_db()

# Утилиты
def embed_image_pil(img_pil):
    """Возвращает L2-нормализованный embedding или None (если лицо не найдено)."""
    try:
        aligned = mtcnn(img_pil)
    except Exception as e:
        st.error(f"MTCNN error: {e}")
        return None
    if aligned is None:
        return None
    with torch.no_grad():
        emb = resnet(aligned.unsqueeze(0))[0].numpy()
    emb = emb / np.linalg.norm(emb)
    return emb

def identify_with_face_db(query_emb, db, top_k=3):
    """Возврат топ-k (person_id, score) отсортированных по убыванию score (cosine)."""
    if not db:
        return []
    names = []
    cents = []
    for k,v in db.items():
        names.append(k)
        cents.append(np.asarray(v))
    cents = np.stack(cents, axis=0)  # (N,512)
    sims = cents @ query_emb  # (N,)
    idxs = np.argsort(-sims)[:top_k]
    return [(names[i], float(sims[i])) for i in idxs]

def identify_with_embeddings(query_emb, X, y, top_k=3):
    if X is None or y is None:
        return []
    sims = X @ query_emb  # shape (N,)
    idxs = np.argsort(-sims)[:top_k]
    return [(str(y[i]), float(sims[i])) for i in idxs]

def save_face_db(db, path="face_db.pkl"):
    with open(path,'wb') as f:
        pickle.dump(db, f)

# UI
st.title("Face Recognition")
st.write("Enroll: добавь пользователя (3+ фото).  Identify: распознай по фото.")

mode = st.sidebar.radio("Mode", ["Identify", "Enroll", "Manage DB"])
threshold = st.sidebar.slider("Порог сходства (cosine)", 0.3, 0.95, eer_threshold, 0.01)
st.sidebar.markdown(f"Авто-порог (EER) = **{eer_threshold:.3f}**")

if mode == "Identify":
    uploaded = st.file_uploader("Загрузить фото для идентификации", type=["jpg","jpeg","png"])
    top_k = st.slider("Показывать топ-k", 1, 5, 3)
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Загруженное изображение", width=320)
        emb = embed_image_pil(img)
        if emb is None:
            st.error("Лицо не найдено — попробуйте другое фото.")
        else:
            st.success("Лицо найдено — вычисляю сходство...")
            results = identify_with_face_db(emb, face_db, top_k)
            if not results and X_emb is not None:
                results = identify_with_embeddings(emb, X_emb, y_labels, top_k)
            if not results:
                st.warning("Нет базы для сравнения.")
            else:
                st.markdown("### Результаты (топ-k)")
                for i, (pid, score) in enumerate(results, start=1):
                    match_text = "MATCH" if score >= threshold else "NO MATCH"
                    st.write(f"**{i}. {pid}** — score={score:.3f} → **{match_text}**")
                best = results[0]
                if best[1] >= threshold:
                    st.success(f"Идентифицирован(а) как **{best[0]}** (score={best[1]:.3f})")
                else:
                    st.info(f"Лучший кандидат: **{best[0]}** (score={best[1]:.3f}) — ниже порога.")
                if st.button("Сохранить результат в лог"):
                    import csv, datetime
                    os.makedirs("logs", exist_ok=True)
                    fn = os.path.join("logs", "recognition_log.csv")
                    with open(fn, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([datetime.datetime.now().isoformat(), best[0], f"{best[1]:.4f}"])
                    st.success("Сохранено в logs/recognition_log.csv")

elif mode == "Enroll":
    st.info("Enrol: введите person_id (например: alice). Загрузите несколько фото (рекомендуется 3+)")
    pid = st.text_input("Person ID")
    files = st.file_uploader("Загрузить изображения", accept_multiple_files=True, type=["jpg","png","jpeg"])
    if st.button("Enroll") and pid and files:
        embs = []
        for f in files:
            img = Image.open(f).convert("RGB")
            emb = embed_image_pil(img)
            if emb is None:
                st.warning(f"Лицо не найдено в {f.name}, пропускаю.")
            else:
                embs.append(emb)
        if not embs:
            st.error("Ни одного корректного лица не найдено.")
        else:
            centroid = np.mean(np.stack(embs, axis=0), axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            face_db[pid] = centroid
            save_face_db(face_db)
            st.success(f"Пользователь {pid} добавлен.")
            st.write(f"Всего в базе: {len(face_db)}")

elif mode == "Manage DB":
    st.write("Управление базой лиц")
    if st.button("Показать список IDs"):
        st.write(list(face_db.keys()))
    if st.button("Очистить базу IDs"):
        if os.path.exists("face_db.pkl"):
            os.remove("face_db.pkl")
        face_db.clear()
        st.success("База очищена.")


st.write("---")
