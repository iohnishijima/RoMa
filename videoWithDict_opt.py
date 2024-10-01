import torch
import cv2
import numpy as np
import pandas as pd
import json
from romatch import roma_outdoor
import tempfile
import os

# パノラマ画像と動画ファイルの設定
eye_tracking_data_df = pd.read_csv('Cockpit-new1.csv', sep=';')
panorama_img_path = 'pano-cockpit-2.jpg'
image_folder = './ScenePics_1/'
json_file = 'cockpit-d850-more.json'
homography_output_path = "./cockpit-new-tinypano.npy"

# RoMaの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
matcher = roma_outdoor(device=device)

# パノラマ画像の読み込み
panorama_img = cv2.imread(panorama_img_path)

# JSONファイルからAOI情報の読み込み
with open(json_file, 'r') as f:
    regions = json.load(f)
    
# カメラパラメータ
camera_matrix = np.array([[500.3346336673587, 0, 286.4275647277839],
                          [0, 497.3594260665638, 251.3027090657917],
                          [0, 0, 1]])

dist_coeffs = np.array([-5.070020e-01, 2.356477e-01, -1.024952e-04, 4.798940e-03, -4.303462e-02])

# 画像フォルダ内のすべての画像ファイルを取得
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('jpeg', 'png', 'jpg'))])

# ホモグラフィを辞書型で保存
homographies = {}

with torch.no_grad():
    for image_file in image_files:
        # 画像ファイルからフレーム番号を取得
        try:
            frame_number = int(''.join(filter(str.isdigit, image_file)))
            print(f"frame_number: {frame_number}, finished reading image file.")
        except ValueError:
            print(f"frame_number: {frame_number}, Failed to read image file.")
            continue

        frame_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"frame_number: {frame_number}, Failed to read image file.")
            continue

        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # 一時ファイルにフレームを保存
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_frame_file:
            cv2.imwrite(tmp_frame_file.name, undistorted_frame)

            # RoMaによる特徴点検出とマッチング（パノラマ画像と一時ファイルのパスを渡す）
            warp, certainty = matcher.match(panorama_img_path, tmp_frame_file.name, device=device)

        try:
            matches, certainty = matcher.sample(warp, certainty)
        except Exception as e:
            print(f"frame_number: {frame_number}, サンプリング中にエラー: {e}")
            homographies[frame_number] = None
            continue

        keypoints_0, keypoints_1 = matcher.to_pixel_coordinates(
            matches, panorama_img.shape[0], panorama_img.shape[1], undistorted_frame.shape[0], undistorted_frame.shape[1]
        )

        if len(keypoints_0) > 8:  # 十分なマッチング点がある場合
            H, mask = cv2.findHomography(np.array(keypoints_1.cpu(), dtype=np.float32), np.array(keypoints_0.cpu(), dtype=np.float32), cv2.RANSAC, 5.0)
            homographies[frame_number] = H
        else:
            homographies[frame_number] = None
            
        transformed_frame = cv2.warpPerspective(undistorted_frame, H, (panorama_img.shape[1], panorama_img.shape[0]))

        # フレームを半透明でパノラマに重ねる
        alpha = 0.5
        panorama_with_gaze = panorama_img.copy()
        cv2.addWeighted(transformed_frame, alpha, panorama_with_gaze, 1 - alpha, 0, panorama_with_gaze)
        cv2.imshow("Panorama with gaze", panorama_with_gaze)
        cv2.waitKey(1)

        print(f"Processed frame {frame_number}")

# ホモグラフィ行列を辞書として保存
np.save(homography_output_path, homographies, allow_pickle=True)
print("ホモグラフィ行列が辞書として保存されました。")
