import os
import numpy as np
import scipy.io as scio

# =========================
# 路径配置
# =========================
ROOT = "/home/yuebai/Data/Dataset/CrossModel/wikipedia_dataset"   # 改成你的真实路径
save_dir = "/home/yuebai/Data/Dataset/CrossModel/WIKI"
os.makedirs(save_dir, exist_ok=True)
IMG_ROOT = os.path.join(ROOT, "images")
TXT_ROOT = os.path.join(ROOT, "texts")

LIST_FILES = [
    os.path.join(ROOT, "trainset_txt_img_cat.list"),
    os.path.join(ROOT, "testset_txt_img_cat.list"),
]

CATEGORIES = [
    "art",
    "biology",
    "geography",
    "history",
    "literature",
    "media",
    "music",
    "royalty",
    "sport",
    "warfare"
]

cat2idx = {c: i for i, c in enumerate(CATEGORIES)}
NUM_CLASS = len(CATEGORIES)

def load_xml_text(xml_path):
    with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()

    start = data.find("<text>")
    end = data.find("</text>")

    if start == -1 or end == -1:
        return ""

    text = data[start + len("<text>"): end]
    text = text.replace("\n", " ").strip()
    return text

# =========================
# 主逻辑
# =========================
index_list = []
caption_list = []
label_list = []

for list_file in LIST_FILES:
    with open(list_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # ===== 常见格式：text_id image_id category =====
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Unexpected format: {line}")

            txt_id, img_id, cat = parts

            cats = CATEGORIES[int(cat)-1]
            # print(cats)

            # 去掉后缀（如果有）
            txt_file = txt_id if txt_id.endswith(".xml") else txt_id + ".xml"
            img_file = img_id if img_id.endswith(".jpg") else img_id + ".jpg"

            # 构造路径
            txt_path = os.path.join(TXT_ROOT, txt_file)
            img_path = os.path.join(IMG_ROOT, cats, img_file)

            if not os.path.exists(txt_path):
                print("continue txt_path:", txt_path)
                continue
            if not os.path.exists(img_path):
                print("continue img_path", img_path)
                continue
            # caption
            # print(txt_path)
            text = load_xml_text(txt_path)

            # label（单标签 one-hot）
            label = np.zeros(NUM_CLASS, dtype=np.float32)
            # label[cat2idx[cat]] = 1.0
            label[int(cat)-1] = 1.0

            index_list.append(img_path)
            caption_list.append([text])
            label_list.append(label)

print("Total samples:", len(index_list))

# =========================
# 保存 mat
# =========================
scio.savemat(
    os.path.join(save_dir, "index.mat"),
    {"index": index_list}
)

scio.savemat(
    os.path.join(save_dir, "caption.mat"),
    {"caption": caption_list}
)

scio.savemat(
    os.path.join(save_dir, "label.mat"),
    {"category": label_list}
)

print("Saved index.mat / caption.mat / label.mat")
