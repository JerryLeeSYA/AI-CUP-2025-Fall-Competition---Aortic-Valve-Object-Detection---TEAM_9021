from ultralytics import YOLO
import os
import shutil
import zipfile
from pathlib import Path
from collections import defaultdict
import numpy as np

# =========================
# 可調參數區
# =========================

# --- Soft-NMS 相關 ---
SOFT_NMS_SIGMA = 0.35         # Gaussian Soft-NMS 的 sigma (越小壓制越強)
SOFT_NMS_SCORE_THR = 1e-4     # Soft-NMS 之後保留框的最小分數

# --- 輸出控制 ---
TOP_K_PER_IMAGE = 10          # 每張圖最多保留幾個框

# --- 模型與路徑 ---
MODEL_PATH = r'runs\detect\train\weights\bset.pt'

# =========================
# Soft-NMS 實作
# =========================

def soft_nms_numpy(boxes, scores, sigma=0.5, score_thr=1e-4):
    """
    Gaussian Soft-NMS 實作
    boxes: (N, 4) [x1, y1, x2, y2]
    scores: (N,)
    回傳: 篩選後的 boxes, scores, indices
    """
    if len(boxes) == 0:
        return boxes, scores, np.array([], dtype=int)

    boxes = boxes.astype(np.float32).copy()
    scores = scores.astype(np.float32).copy()
    N = boxes.shape[0]
    indices = np.arange(N)

    for i in range(N):
        # 1. 找出目前 i 以後分數最高的框，交換到第 i 個位置
        max_pos = i + np.argmax(scores[i:])
        if scores[max_pos] > scores[i]:
            boxes[[i, max_pos]] = boxes[[max_pos, i]]
            scores[[i, max_pos]] = scores[[max_pos, i]]
            indices[[i, max_pos]] = indices[[max_pos, i]]

        # 2. 計算與第 i 個框的 IoU
        x1, y1, x2, y2 = boxes[i]

        xx1 = np.maximum(x1, boxes[i+1:, 0])
        yy1 = np.maximum(y1, boxes[i+1:, 1])
        xx2 = np.minimum(x2, boxes[i+1:, 2])
        yy2 = np.minimum(y2, boxes[i+1:, 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        area_i = (x2 - x1) * (y2 - y1)
        area_rest = (boxes[i+1:, 2] - boxes[i+1:, 0]) * (boxes[i+1:, 3] - boxes[i+1:, 1])
        union = area_i + area_rest - inter + 1e-6
        iou = inter / union

        # 3. Gaussian 衰減
        weight = np.exp(-(iou ** 2) / sigma)
        scores[i+1:] *= weight

    # 4. 篩掉太小的分數
    keep = np.where(scores > score_thr)[0]
    return boxes[keep], scores[keep], indices[keep]


# =========================
# 1. 準備路徑與資料
# =========================

zip_file_name = 'testing_image.zip'
unzip_folder = 'testing_image_unzipped'
predict_source_dir = Path('./datasets/test/images')

if predict_source_dir.exists() and any(predict_source_dir.iterdir()):
    print(f"'{predict_source_dir}' 已存在且包含檔案，跳過資料準備步驟。")
else:
    if not os.path.exists(zip_file_name):
        print(f"錯誤：找不到 '{zip_file_name}'。")
    else:
        print(f"找到 '{zip_file_name}'，開始解壓縮...")
        os.makedirs(unzip_folder, exist_ok=True)
        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            zip_ref.extractall(unzip_folder)
        print(f"已成功解壓縮到 '{unzip_folder}'。")

        source_root = unzip_folder
        for dirpath, dirnames, _ in os.walk(source_root):
            if any(d.lower().startswith("patient") for d in dirnames):
                source_root = dirpath
                break
        print(f"已自動找到 patient 資料夾的根目錄為：'{source_root}'")

        os.makedirs(predict_source_dir, exist_ok=True)

        all_files_to_move = []
        for patient_dir in Path(source_root).iterdir():
            if patient_dir.is_dir() and patient_dir.name.lower().startswith("patient"):
                for image_path in patient_dir.glob('*.*'):
                    if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        all_files_to_move.append(str(image_path))

        print(f"找到 {len(all_files_to_move)} 個圖片檔案，開始移動...")
        for file_path in all_files_to_move:
            destination_path = os.path.join(predict_source_dir, os.path.basename(file_path))
            shutil.move(file_path, destination_path)
        print(f"移動完成！所有檔案已移動至：'{predict_source_dir}'")

        try:
            shutil.rmtree(unzip_folder)
            print(f"已成功刪除暫存資料夾 '{unzip_folder}'。")
        except OSError as e:
            print(f"刪除暫存資料夾時出錯: {e}")


# =========================
# 2. 載入模型並執行預測
# =========================

print("\n--- 開始執行模型預測 ---")
model = YOLO(MODEL_PATH)

results = list(model.predict(
    source=predict_source_dir,
    save=False,
    imgsz=640,
    conf=0.0005,     # 極低閾值，讓 Soft-NMS 有更多候選框
    iou=0.55,         # 幾乎禁用內建 NMS，交給 Soft-NMS 處理
    augment=True,    # TTA
    batch=30,
    device=[0, 1, 2],
))


# =========================
# 3. 收集所有預測
# =========================

per_image_preds = defaultdict(list)

for result in results:
    filename = Path(result.path).stem
    boxes = result.boxes

    if len(boxes) == 0:
        continue

    for i in range(len(boxes)):
        label = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()

        per_image_preds[filename].append({
            "bbox": [x1, y1, x2, y2],
            "score": conf,
            "label": label,
        })

print(f"\n共收集到 {len(per_image_preds)} 張圖片的預測結果。")


# =========================
# 4. Soft-NMS 後處理
# =========================

print(f"\n--- 執行 Soft-NMS (sigma={SOFT_NMS_SIGMA}) ---")
output_dir = './predict_txt'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'images_softnms.txt')

total_boxes = 0

with open(output_path, 'w', encoding='utf-8') as f_out:
    for filename, preds in per_image_preds.items():
        if len(preds) == 0:
            continue

        boxes = np.array([p["bbox"] for p in preds], dtype=np.float32)
        scores = np.array([p["score"] for p in preds], dtype=np.float32)
        labels = np.array([p["label"] for p in preds], dtype=int)

        # Soft-NMS
        boxes_s, scores_s, keep_idx = soft_nms_numpy(
            boxes, scores,
            sigma=SOFT_NMS_SIGMA,
            score_thr=SOFT_NMS_SCORE_THR
        )
        labels_s = labels[keep_idx]

        # 依分數排序
        order = np.argsort(scores_s)[::-1]
        boxes_s = boxes_s[order]
        scores_s = scores_s[order]
        labels_s = labels_s[order]

        # 每張圖最多保留 TOP_K_PER_IMAGE 個框
        num_keep = min(TOP_K_PER_IMAGE, len(boxes_s))
        total_boxes += num_keep

        for i in range(num_keep):
            x1, y1, x2, y2 = boxes_s[i]
            conf = scores_s[i]
            label = int(labels_s[i])
            line = f"{filename} {label} {conf:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
            f_out.write(line)

print(f"\n✅ 結果已寫入：{output_path}")
print(f"總框數: {total_boxes}")
print(f"平均每張圖框數: {total_boxes / len(per_image_preds):.2f}")
print("\n🚀 完成！請用 Smart Filter 處理後再提交")