from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
import torch
from pathlib import Path

model = InceptionResnetV1(pretrained='vggface2').eval()

def get_emb_from_path(path):
    img = Image.open(path).convert('RGB').resize((160,160))
    arr = (np.asarray(img).astype('float32') - 127.5) / 128.0
    t = torch.tensor(arr).permute(2,0,1).unsqueeze(0)
    with torch.no_grad():
        emb = model(t)[0].cpu().numpy()
    emb = emb / np.linalg.norm(emb)
    return emb

def extract_folder(folder):
    folder = Path(folder)
    X, y, paths = [], [], []
    for person in [p for p in folder.iterdir() if p.is_dir()]:
        for img in person.glob('*'):
            try:
                e = get_emb_from_path(str(img))
                X.append(e); y.append(person.name); paths.append(str(img))
            except Exception as ex:
                print("skip", img, ex)
    return np.stack(X,0), np.array(y), paths