"""
主訓練腳本 - 針對樣本平衡後的數據集
使用RandAugment + 固定的強增強策略
"""
import os
import sys
import multiprocessing
from pathlib import Path
from ultralytics import YOLO

def run_training():
    """執行訓練"""
    
    # ========== 配置 ==========
    model_cfg = None
    pretrained_weights = 'yolo12l.pt'
    
    # Resume 設置
    checkpoint_path = 'runs/detect/train/weights/last.pt'
    
    # ========== 決定訓練模式 ==========
    
    if Path(checkpoint_path).exists():
        # 🔄 恢復訓練模式
        print("=" * 60)
        print("🔄 檢測到 Checkpoint，進入恢復訓練模式")
        print(f"   Checkpoint: {checkpoint_path}")
        print("=" * 60)
        
        # ✅ 正確做法：直接從 checkpoint 加載
        model = YOLO(checkpoint_path)
        
        # ✅ 直接 resume，不需要其他參數
        results = model.train(resume=True)
        
    else:
        # 🆕 全新訓練模式
        print("=" * 60)
        print("🆕 開始全新訓練")
        print(f"   架構: {model_cfg}")
        print(f"   預訓練: {pretrained_weights}")
        print("=" * 60)
        
        if model_cfg is None:
            # ✅ 直接用 .pt 建立模型（不用 model_cfg）
            print(f"✅ 從預訓練權重建立模型: {pretrained_weights}")
            model = YOLO(pretrained_weights)
        else:
            # 創建新模型
            model = YOLO(model_cfg)
            # 加載預訓練權重（可選）
            if Path(pretrained_weights).exists():
                print(f"✅ 加載預訓練權重: {pretrained_weights}")
                model.load(pretrained_weights)
        
        # 完整訓練配置
        results = model.train(
            # 基本配置
            data="./aortic_valve_colab.yaml",
            epochs=150,
            # yolo12l
            batch=24,
            # yolo12x
            # batch=18,
            device=[0, 1, 2],
            workers=24,
            cache='disk',
            patience=50,
            seed=42,
            deterministic=True,
            
            # ⚠️ 關鍵：全新訓練的設置
            resume=False,
            exist_ok=False,  # 不覆蓋，自動遞增名稱
            
            # 優化器配置
            optimizer='auto',
            lr0=0.008,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.001,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.0,
            
            # 增強策略
            augment=True,
            auto_augment='randaugment',
            mosaic=0.6,
            mixup=0.2,
            copy_paste=0.0,
            
            # 幾何增強
            degrees=5.0,
            translate=0.1,
            scale=0.3,
            shear=3.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            
            # 顏色增強
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            bgr=0.0,
            erasing=0.4,
            
            # 訓練策略
            label_smoothing=0.0,
            dropout=0.1,
            amp=True,
            cos_lr=True,
            close_mosaic=20,
            
            # 驗證和保存
            val=True,
            save=True,
            save_period=5,
            plots=True,
            verbose=True,
            
            # 後處理
            conf=0.001,
            iou=0.7,
            max_det=300,
            
            # 其他
            single_cls=False,
            rect=False,
            overlap_mask=True,
            mask_ratio=4,
            profile=False,
            freeze=None,
            multi_scale=False,
        )
    
    print("\n✅ 訓練完成！")
    return results

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    try:
        results = run_training()
    except KeyboardInterrupt:
        print("\n⚠️  訓練被用戶中斷")
    except Exception as e:
        print(f"\n❌ 訓練失敗: {e}")
        import traceback
        traceback.print_exc()