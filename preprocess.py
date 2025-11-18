from facenet_pytorch import MTCNN
from PIL import Image
from pathlib import Path
import os
import torch

mtcnn = MTCNN(image_size=160, margin=14, keep_all=False)

def align_and_save(src_path, dst_path):
    """Загружает изображение и конвертирует в RGB. Передаёт его в MTCNN"""
    img = Image.open(src_path).convert('RGB')
    aligned = mtcnn(img)  # tensor [3,160,160] or None
    if aligned is None:
        return False
    if isinstance(aligned, torch.Tensor):
        arr = aligned.permute(1,2,0).cpu().numpy()  # shape H,W,3, values ~[-1,1]
        arr = (arr * 127.5 + 127.5).clip(0,255).astype('uint8')
        Image.fromarray(arr).save(dst_path)
    else:
        aligned.save(dst_path)
    Image.fromarray(arr).save(dst_path)
    return True

def run_on_folder(src_root, dst_root):
    """Преобразует пути к папкам в объекты Path для удобной работы."""
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    for person in [p for p in src_root.iterdir() if p.is_dir()]:
        out_dir = dst_root / person.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in person.glob('*'):
            if img.suffix.lower() not in ('.jpg','.jpeg','.png'):
                continue
            ok = align_and_save(str(img), str(out_dir / img.name))
            if not ok:
                print("No face:", img)

if __name__ == '__main__':

    import sys
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py <src_root> <dst_root>")
    else:
        run_on_folder(sys.argv[1], sys.argv[2])