import cv2
import numpy as np
import pandas as pd
import json
from PIL import Image
import os

OverlayVieo = True

# パノラマ画像と動画ファイルの設定
eye_tracking_data_df = pd.read_csv('Cockpit-new1.csv', sep=';')
panorama_img_path = 'try-plz\pano-cockpit-2.jpg'  # 画像パスに修正
image_folder = './ScenePics_1/'
json_file = 'cockpit-d850-more-outiside.json'
homography_output_path = "try-plz\cockpit2_pano-cockpit-2.npy"

loaded_homographies = np.load(homography_output_path, allow_pickle=True).item()
print(len(loaded_homographies))

# パノラマ画像の読み込み
panorama_img = Image.open(panorama_img_path)
panorama_img_np = np.array(panorama_img)

# JSONファイルからAOI情報の読み込み
with open(json_file, 'r') as f:
    regions = json.load(f)

# カメラパラメータ
camera_matrix = np.array([[500.3346336673587, 0, 286.4275647277839],
                          [0, 497.3594260665638, 251.3027090657917],
                          [0, 0, 1]])

dist_coeffs = np.array([-5.070020e-01, 2.356477e-01, -1.024952e-04, 4.798940e-03, -4.303462e-02])

# 視線座標の履歴を保持するリスト
gazeX_history = []
gazeY_history = []

# 視線座標をフィルタリングする関数
def get_filtered_gaze(rawGazeX, rawGazeY):
    gazeX_history.append(rawGazeX)
    gazeY_history.append(rawGazeY)

    max_history_length = 5
    if len(gazeX_history) > max_history_length:
        gazeX_history.pop(0)
    if len(gazeY_history) > max_history_length:
        gazeY_history.pop(0)

    filteredGazeX = sum(gazeX_history) / len(gazeX_history)
    filteredGazeY = sum(gazeY_history) / len(gazeY_history)
    return int(filteredGazeX), int(filteredGazeY)

# CSVファイルに保存するためのデータフレームを初期化
gaze_data_output = []

output_frames = []

def calculate_gaze_size(fCount, start_size, max_size, max_count):
    size_ratio = min(fCount / max_count, 1)
    return int(start_size + (max_size - start_size) * size_ratio)

fFlag = False
fCount = 0
start_size = 5
max_size = 25
initial_alpha = 0.25
alpha_step = 0.03
max_count = 40
total_gaze_points = len(eye_tracking_data_df)

image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('jpeg', 'png'))])

for index, row in eye_tracking_data_df.iterrows():
    print(f"process frame {index}/{total_gaze_points}")
    gaze_x = float(row.iloc[2])
    gaze_y = float(row.iloc[3])
    frameEye = index
    RScore = float(row.iloc[9])
    LScore = float(row.iloc[10])
    eyeEvent = str(row.iloc[20])
    frame_number = int(row.iloc[0])
    image_file = f"frame_{frame_number}.jpeg"
    
    try:
        frame = cv2.imread(os.path.join(image_folder, image_file))
        H = loaded_homographies.get(frame_number, None)
        if H is not None:
            gaze_point = np.array([[gaze_x * 640, gaze_y * 480]], dtype='float32')
            gaze_point = np.array([gaze_point])
            transformed_gaze_point = cv2.perspectiveTransform(gaze_point, H)[0][0]

            filtered_gaze_x, filtered_gaze_y = get_filtered_gaze(transformed_gaze_point[0], transformed_gaze_point[1])

            panorama_with_gaze = panorama_img_np.copy()
            gaze_inside_any_polygon = False
            inside_polygon_label = "NA"

            if OverlayVieo:
                # ビデオフレームをホモグラフィーで変換してパノラマ画像に重ねる
                undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
                transformed_frame = cv2.warpPerspective(undistorted_frame, H, (panorama_with_gaze.shape[1], panorama_with_gaze.shape[0]))
                print(f"transformed_frame dtype: {transformed_frame.dtype}")
                print(f"panorama_with_gaze dtype: {panorama_with_gaze.dtype}")
                
                if panorama_with_gaze.shape[2] == 4:
                    panorama_with_gaze = panorama_with_gaze[:, :, :3]  # アルファチャンネルを取り除く
                    print("Removed alpha channel from panorama_with_gaze.")

                # フレームを半透明でパノラマに重ねる
                alpha = 0.5
                cv2.addWeighted(transformed_frame, alpha, panorama_with_gaze, 1 - alpha, 0, panorama_with_gaze)

            # # 視線の描画やその他の処理
            for label, points in regions.items():
                scaled_points = [{'x': point['x'] * 0.2, 'y': point['y'] * 0.2} for point in points]
                pts = np.array([[point["x"], point["y"]] for point in scaled_points], np.int32)
                pts = pts.reshape((-1, 1, 2))

                overlay = panorama_with_gaze.copy()

                if cv2.pointPolygonTest(pts, (filtered_gaze_x, filtered_gaze_y), False) >= 0:
                    color = (0, 0, 255)
                    gaze_inside_any_polygon = True
                    inside_polygon_label = label
                    alpha = 0.5
                else:
                    color = (0, 255, 0)
                    alpha = 0.1

                cv2.fillPoly(overlay, [pts], color=color)

                center_x = int(np.mean([point["x"] for point in points]))
                center_y = int(np.mean([point["y"] for point in points]))

                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

                text_x = center_x - text_size[0] // 2
                text_y = center_y

                cv2.putText(overlay, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.addWeighted(overlay, alpha, panorama_with_gaze, 1 - alpha, 0, panorama_with_gaze)

            if eyeEvent[:2] == "FB":
                fFlag = True

            if fFlag == True:
                fCount += 1

            overlay2 = panorama_with_gaze.copy()
            gaze_size = calculate_gaze_size(fCount, start_size, max_size, max_count)
            current_alpha = min(initial_alpha + (fCount * alpha_step), 1)
            cv2.circle(overlay2, (filtered_gaze_x, filtered_gaze_y), gaze_size, (0, 0, 255,), -1)
            cv2.addWeighted(overlay2, current_alpha, panorama_with_gaze, 1 - current_alpha, 0, panorama_with_gaze)
            cv2.circle(panorama_with_gaze, (filtered_gaze_x, filtered_gaze_y), gaze_size, (0, 0, 255), 2)

            if eyeEvent[:2] == "FE":
                fCount = 0
                fFlag = False

            output_frames.append(panorama_with_gaze)

            gaze_data_output.append({
                "frame": frameEye,
                "gazeX": filtered_gaze_x,
                "gazeY": filtered_gaze_y,
                "RScore": RScore,
                "LScore": LScore,
                "eyeEvent": eyeEvent,
                "inside_polygon": inside_polygon_label
            })
        else:
            panorama_without_gaze = panorama_with_gaze.copy()
            output_frames.append(panorama_without_gaze)
            gaze_data_output.append({
                "frame": frameEye,
                "gazeX": None,
                "gazeY": None,
                "RScore": RScore,
                "LScore": LScore,
                "eyeEvent": eyeEvent,
                "inside_polygon": "NA"
            })

    except IndexError:
        break
    
    if index == len(loaded_homographies):
        break

# 動画の保存
height, width, layers = output_frames[0].shape
output_video_path = "./output_video_RoMa_withoutVideoFrame_Cockpit_withFrame_comeon.webm"
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'VP80'), 30, (width, height))

for frame in output_frames:
    out.write(frame)

out.release()

# CSVデータの保存
gaze_df = pd.DataFrame(gaze_data_output)
output_csv_path = "./annotated_gaze_data_RoMa_withFrame_comeon_Cockpit.csv"
gaze_df.to_csv(output_csv_path, index=False)

print("動画とCSVが保存されました")