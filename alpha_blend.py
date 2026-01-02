import cv2
import numpy as np
import os
import argparse
from natsort import natsorted

def main():
    parser = argparse.ArgumentParser(description="Alpha blending: RGB from foreground, alpha = max(fg, base)")
    parser.add_argument("input_dir", help="前景圖片資料夾")
    parser.add_argument("base_path", help="背景圖片（用於 alpha blend）")
    parser.add_argument("output_dir", help="輸出資料夾")
    parser.add_argument("-r", "--rootpath", default="", help="根路徑，可為空，若有則會自動附加到所有路徑前面")
    args = parser.parse_args()

    # 加上 root path（若有）
    def join_root(p):
        return os.path.join(args.rootpath, p) if args.rootpath else p

    input_dir = join_root(args.input_dir)
    base_path = join_root(args.base_path)
    output_dir = join_root(args.output_dir)

    # 建立輸出資料夾
    os.makedirs(output_dir, exist_ok=True)

    # 讀取背景
    base = cv2.imread(base_path, cv2.IMREAD_UNCHANGED)
    if base is None:
        raise FileNotFoundError(f"找不到背景圖片：{base_path}")

    # 取背景 alpha
    if base.shape[2] < 4:
        base_alpha = np.ones(base.shape[:2], dtype=np.uint8) * 255
    else:
        base_alpha = base[:, :, 3]

    # 取得前景清單（自然排序）
    files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    files = natsorted(files)
    print(f"🖼️  共找到 {len(files)} 張圖片，開始處理…")

    for idx, fname in enumerate(files, 1):
        fpath = os.path.join(input_dir, fname)
        fg = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if fg is None:
            print(f"[警告] 無法讀取：{fname}，跳過。")
            continue

        # 確保尺寸一致
        if fg.shape[:2] != base.shape[:2]:
            fg = cv2.resize(fg, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_AREA)

        # 前景 alpha
        if fg.shape[2] < 4:
            fg_alpha = np.ones(fg.shape[:2], dtype=np.uint8) * 255
        else:
            fg_alpha = fg[:, :, 3]

        # α = max(fg_alpha, base_alpha)
        merged_alpha = np.maximum(fg_alpha, base_alpha)

        # RGB = 前景
        rgb = fg[:, :, :3]

        # 合併 RGBA
        result = np.dstack((rgb, merged_alpha))

        # 輸出
        out_path = os.path.join(output_dir, f"frame_{idx:04d}.png")
        cv2.imwrite(out_path, result)
        print(f"[{idx:3d}/{len(files)}] {fname} -> {os.path.basename(out_path)}")

    print("✅ 全部處理完畢！")

if __name__ == "__main__":
    main()
