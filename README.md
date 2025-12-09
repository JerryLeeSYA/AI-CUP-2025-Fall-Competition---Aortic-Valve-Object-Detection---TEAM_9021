# 2025 AI CUP Autumn - CT Aortic Valve Object Detection
# 電腦斷層主動脈瓣物件偵測競賽 - 解決方案

本專案為 **2025 AI CUP 秋季賽** 的解決方案。我們採用 **YOLOv12-Large** 架構，結合 **正樣本過採樣 (Oversampling)** 解決資料不平衡問題，並在推論階段導入 **TTA (Test Time Augmentation)** 與 **Gaussian Soft-NMS** 技術。

🏆 **競賽成績 (Private Leaderboard)**:
* **Score (mAP)**: 0.964516
* **Rank**: 36

## 📋 目錄 / Table of Contents
1. [專案架構 (Project Structure)](#-專案架構-project-structure)
2. [環境需求 (Requirements)](#-環境需求-requirements)
3. [安裝與設定 (Installation)](#-安裝與設定-installation)
4. [執行流程 (Usage)](#-執行流程-usage)
    - [Step 1: 資料前處理](#step-1-資料前處理-data-preprocessing)
    - [Step 2: 模型訓練](#step-2-模型訓練-training)
    - [Step 3: 預測與後處理](#step-3-預測與後處理-inference--soft-nms)
5. [方法論概述 (Methodology)](#-方法論概述-methodology)

## 📂 專案架構 (Project Structure)

在開始執行之前，請確認您的目錄結構如下，並將官方提供的三個 ZIP 檔放置於根目錄：

Project_Root/
├── 1_get_dataset.py        # 資料清洗與過採樣腳本
├── 2_train.py              # YOLOv12 訓練腳本 (支援 DDP)
├── 3_predict_softNMS.py    # 推論腳本 (含 Soft-NMS & TTA)
├── aortic_valve_colab.yaml # 資料集設定檔
├── requirements.txt        # Python 依賴套件清單
├── yolo12l.pt              # 預訓練權重 (首次執行會自動下載)
├── training_image.zip      # [官方原始資料]
├── training_label.zip      # [官方原始資料]
└── testing_image.zip       # [官方原始資料]

## 💻 環境需求 (Requirements)
本專案測試於 Windows 10 環境。

OS: Windows 10 / 11 (亦支援 Linux)

Python: 3.9+

GPU: 建議 NVIDIA RTX 2080Ti x3

CUDA: 11.8+

## ⚙️ 安裝與設定 (Installation)
請依序執行以下指令來建立虛擬環境並安裝依賴套件：

# 1. Clone 本專案
git clone [您的 Github Repo 連結]
cd [專案資料夾名稱]

# 2. 建立 Python 虛擬環境
python -m venv .venv

# 3. 啟動虛擬環境 (Windows)
.\.venv\Scripts\activate
# (若是 Linux/Mac 請使用: source .venv/bin/activate)

# 4. 安裝必要套件
pip install -r requirements.txt

## 🚀 執行流程 (Usage)
請依照順序執行以下三個 Python 腳本。

Step 1: 資料前處理 (Data Preprocessing)
解壓縮原始資料，並進行 5 倍正樣本過採樣 (Oversampling)。
python 1_get_dataset.py

輸入: training_image.zip, training_label.zip
輸出: ./datasets 資料夾 (包含 train/val 分割)
功能:
自動遞迴搜尋 patient 資料夾。
Patient-Level Split: Patient 01-40 (Train), 41-50 (Val)。
Oversampling: 正樣本複製 5 份 (_aug_0 ~ _aug_4) 以平衡正負樣本比例。

Step 2: 模型訓練 (Training)
執行 YOLOv12-Large 模型訓練。
python 2_train.py

設定: 讀取 aortic_valve_colab.yaml。
參數:
    Epochs: 150
    Batch Size: 24 (依據 3x 2080Ti 設定)
    Augmentation: RandAugment, Mosaic(0.6), Mixup(0.2)
輸出: 訓練權重將儲存於 runs/detect/train/weights/best.pt。

Step 3: 預測與後處理 (Inference & Soft-NMS)
執行預測並生成最終提交檔案。
python 3_predict_softNMS.py

技術:
啟用 TTA (Test Time Augmentation)。
使用 Gaussian Soft-NMS (Sigma=0.35, Score Thr=1e-4)。
每張圖保留 Top-10 預測框。
最終結果: 檔案位於 predict_txt/images_softnms.txt (可直接上傳競賽系統)。

## 🧠 方法論概述 (Methodology)
資料平衡 (Data Balancing): 透過 5 倍過採樣顯著提升模型對主動脈瓣的 Recall。
強增強訓練 (Strong Augmentation): 使用 Mosaic 與 Mixup 加上幾何變換，提升模型泛化能力。
Gaussian Soft-NMS: 不同於傳統 NMS 直接刪除重疊框，Soft-NMS 透過高斯函數衰減重疊框的分數，在 Recall 與 Precision 之間取得最佳平衡，特別適用於 IoU 門檻較高的評測。

Author: TEAM_9021