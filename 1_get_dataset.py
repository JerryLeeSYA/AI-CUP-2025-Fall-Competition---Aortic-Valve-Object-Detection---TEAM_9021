import os
import shutil
from pathlib import Path
from tqdm import tqdm
import zipfile

# --- 1. 設定 ---
train_img_zip = 'training_image.zip'
train_lbl_zip = 'training_label.zip'
temp_unzip_path = Path('./temp_unzip_for_prep')
final_dataset_path = Path('./datasets')
train_patient_range = range(1, 40)
val_patient_range = range(41, 50)
oversample_factor = 5  # <--- 在這裡設定您想要的過採樣倍數

# --- 2. 輔助函式 ---
def find_patient_root(root):
    """智能查找包含 'patient...' 系列子資料夾的實際根目錄"""
    try:
        if any(d.lower().startswith("patient") for d in os.listdir(root)): return Path(root)
    except FileNotFoundError: return Path(root)
    for dirpath, dirnames, _ in os.walk(root):
        if any(d.lower().startswith("patient") for d in dirnames): return Path(dirpath)
    return Path(root)

# --- 3. 主程式 ---
def main():
    print("--- 開始建立資料集 ---")
    
    # 清理舊資料夾
    train_path = final_dataset_path / 'train'
    val_path = final_dataset_path / 'val'
    print("正在清理舊的 train 和 val 資料夾...")
    if train_path.exists(): shutil.rmtree(train_path)
    if val_path.exists(): shutil.rmtree(val_path)
    if temp_unzip_path.exists(): shutil.rmtree(temp_unzip_path)

    # 建立目標資料夾結構
    (train_path / 'images').mkdir(parents=True, exist_ok=True)
    (train_path / 'labels').mkdir(parents=True, exist_ok=True)
    (val_path / 'images').mkdir(parents=True, exist_ok=True)
    (val_path / 'labels').mkdir(parents=True, exist_ok=True)

    # 解壓縮
    print("正在解壓縮原始檔案...")
    with zipfile.ZipFile(train_img_zip, 'r') as z: z.extractall(temp_unzip_path / 'images')
    with zipfile.ZipFile(train_lbl_zip, 'r') as z: z.extractall(temp_unzip_path / 'labels')
    
    img_root = find_patient_root(temp_unzip_path / 'images')
    lbl_root = find_patient_root(temp_unzip_path / 'labels')
    
    # 處理資料複製
    print(f"處理中... 訓練集正樣本將被複製 {oversample_factor} 次。")
    # 處理訓練集
    for i in tqdm(train_patient_range, desc="正在處理訓練集"):
        patient_id = f"patient{i:04d}"
        patient_img_dir = img_root / patient_id
        if not patient_img_dir.is_dir(): continue
        for img_path in patient_img_dir.glob('*.*'):
            if not img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']: continue
            lbl_path = lbl_root / patient_id / f"{img_path.stem}.txt"
            if lbl_path.exists(): # 正樣本
                for j in range(oversample_factor):
                    new_img_name = f"{img_path.stem}_aug_{j}{img_path.suffix}"
                    new_lbl_name = f"{img_path.stem}_aug_{j}.txt"
                    shutil.copy(img_path, train_path / 'images' / new_img_name)
                    shutil.copy(lbl_path, train_path / 'labels' / new_lbl_name)
            else: # 負樣本
                shutil.copy(img_path, train_path / 'images' / img_path.name)

    # 處理驗證集
    for i in tqdm(val_patient_range, desc="正在處理驗證集"):
        patient_id = f"patient{i:04d}"
        patient_img_dir = img_root / patient_id
        if not patient_img_dir.is_dir(): continue
        for img_path in patient_img_dir.glob('*.*'):
            if not img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']: continue
            lbl_path = lbl_root / patient_id / f"{img_path.stem}.txt"
            shutil.copy(img_path, val_path / 'images' / img_path.name)
            if lbl_path.exists():
                shutil.copy(lbl_path, val_path / 'labels' / lbl_path.name)

    # 清理暫存檔
    print("正在清理暫存檔案...")
    shutil.rmtree(temp_unzip_path)

    # --- 最終統計 (直接從新產生的資料夾統計) ---
    print("\n--- 新資料集統計結果 ---")
    train_final_images = list((train_path / 'images').glob('*.*'))
    train_final_labels = list((train_path / 'labels').glob('*.txt'))
    train_pos_count = len(train_final_labels)
    train_neg_count = len(train_final_images) - train_pos_count
    
    val_final_images = list((val_path / 'images').glob('*.*'))
    val_final_labels = list((val_path / 'labels').glob('*.txt'))
    val_pos_count = len(val_final_labels)
    val_neg_count = len(val_final_images) - val_pos_count

    print(f"訓練集 ('{train_path}'):")
    print(f"  - 正樣本數 (有標註): {train_pos_count}")
    print(f"  - 負樣本數 (無標註): {train_neg_count}")
    print(f"  - 總圖片數: {len(train_final_images)}")
    if train_final_images: print(f"  - 正樣本比例: {train_pos_count / len(train_final_images) * 100:.2f}%")

    print(f"\n驗證集 ('{val_path}'):")
    print(f"  - 正樣本數 (有標註): {val_pos_count}")
    print(f"  - 負樣本數 (無標註): {val_neg_count}")
    print(f"  - 總圖片數: {len(val_final_images)}")
    if val_final_images: print(f"  - 正樣本比例: {val_pos_count / len(val_final_images) * 100:.2f}%")

if __name__ == '__main__':
    main()