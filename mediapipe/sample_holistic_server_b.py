#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse
import cv2 as cv
import numpy as np
import mediapipe as mp
import math

from utils import CvFpsCalc

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import threading
from time import sleep
import time 

global pitch     #f_ud 　顔の前後傾き
global yaw       # 顔の回転
global roll      #eye_slope　顔の左右傾き
global lb_dist   #lb_dist
global rb_dist   #rb_dist
global m_out     #m_out
global sholder_z #sholder_z
global sholder_x #sholder_x 
global left_eye_value  #left_eye_value
global right_eye_value #right_eye_value
global start_stop

app = FastAPI()

start_stop="start"

pitch =0.0    #f_ud 　顔の前後傾き
yaw   =0.0     # 顔の回転
roll =0.0      #eye_slope　顔の左右傾き
lb_dist =0.0   #lb_dist
rb_dist =0.0   #rb_dist
m_out =0.0     #m_out
sholder_z=0.0  #sholder_z
sholder_x =0.0 #sholder_x 
left_eye_value=0.0   #left_eye_value
right_eye_value=0.0  #right_eye_value

@app.post("/get_mcap/")
def get_mcap(mode: str = Form(...)):
    global pitch     #f_ud 　顔の前後傾き
    global yaw       # 顔の回転
    global roll      #eye_slope　顔の左右傾き
    global lb_dist   #lb_dist
    global rb_dist   #rb_dist
    global m_out     #m_out
    global sholder_z #sholder_z
    global sholder_x #sholder_x 
    global left_eye_value  #left_eye_value
    global right_eye_value #right_eye_value
    global start_stop
    global face_diagonal_3d
    
    start_stop=mode
    print(">>>>>>>>>>>>>>>>>>",mode)
    # 辞書で返す値をまとめる
    return {
        "pitch": pitch/100,
        "yaw": yaw/30,
        "roll": roll/100,
        "lb_dist": lb_dist/100,
        "rb_dist": rb_dist/100,
        "m_out": m_out,
        "sholder_z": sholder_z,
        "sholder_x": sholder_x/10,
        "left_eye_value": left_eye_value,
        "right_eye_value": right_eye_value
    }

def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--device", type=int, default=0)
        parser.add_argument("--width", help='cap width', type=int, default=960)
        parser.add_argument("--height", help='cap height', type=int, default=540)
        parser.add_argument('--unuse_smooth_landmarks', action='store_true')
        parser.add_argument('--enable_segmentation', action='store_true')
        #parser.add_argument('--smooth_segmentation', action='store_true')
        parser.add_argument("--model_complexity",
                            help='model_complexity(0,1(default),2)',
                            type=int,
                            default=1)
        parser.add_argument("--min_detection_confidence",
                            help='face mesh min_detection_confidence',
                            type=float,
                            default=0.5)
        parser.add_argument("--min_tracking_confidence",
                            help='face mesh min_tracking_confidence',
                            type=int,
                            default=0.5)
        parser.add_argument("--segmentation_score_th",
                            help='segmentation_score_threshold',
                            type=float,
                            default=0.5)

        parser.add_argument('--use_brect', action='store_true')
        #parser.add_argument('--plot_world_landmark', action='store_true')

        args = parser.parse_args()

        return args

def init():
        # 引数解析 #################################################################
        args = get_args()
        cap_device = args.device
        cap_width = args.width
        cap_height = args.height
        smooth_landmarks = not args.unuse_smooth_landmarks
        enable_segmentation = args.enable_segmentation
        #smooth_segmentation = args.smooth_segmentation
        model_complexity = args.model_complexity
        min_detection_confidence = args.min_detection_confidence
        min_tracking_confidence = args.min_tracking_confidence
        segmentation_score_th = args.segmentation_score_th
        #plot_world_landmark = args.plot_world_landmark
        # カメラ準備 
        cap = cv.VideoCapture(cap_device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
        # モデルロード
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        # FPS計測モジュール 
        cvFpsCalc = CvFpsCalc(buffer_len=10)
        get_mocap(cvFpsCalc,holistic,cap,segmentation_score_th,enable_segmentation)

def get_mocap(cvFpsCalc,holistic,cap,segmentation_score_th,enable_segmentation):
        global pitch     #f_ud 　顔の前後傾き
        global yaw       # 顔の回転
        global roll      #eye_slope　顔の左右傾き
        global lb_dist   #lb_dist
        global rb_dist   #rb_dist
        global m_out     #m_out
        global sholder_z #sholder_z
        global sholder_x #sholder_x 
        global left_eye_value  #left_eye_value
        global right_eye_value #right_eye_value
        global face_diagonal_3d

        global start_stop
        
        print("****get_mocap****",start_stop)
        while True:
            if start_stop=="start":
                display_fps = cvFpsCalc.get()
                # カメラキャプチャ #####################################################
                ret, image = cap.read()
                if not ret:
                    break
                image = cv.flip(image, 1)  # ミラー表示
                cap_img=copy.deepcopy(image)
                debug_image = copy.deepcopy(image)
                # 検出実施 #############################################################
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                # Face Mesh ###########################################################
                face_landmarks = results.face_landmarks
                if face_landmarks is not None:
                    #landmark_pointの取得と描画
                    debug_image, landmark_point,lb_dist,rb_dist,m_out,left_eye_value,right_eye_value = draw_face_landmarks(debug_image, face_landmarks)

                # Face Meshの3Dランドマークを取得
                '''
                if results.face_landmarks:
                    for landmark in results.face_landmarks.landmark:
                        x = landmark.x  # 相対的なx座標（0.0 - 1.0）
                        y = landmark.y  # 相対的なy座標（0.0 - 1.0）
                        z = landmark.z  # 相対的なz座標（深度情報）
                        
                        # ピクセル座標への変換
                        image_height, image_width, _ = image.shape
                        x_px = int(landmark.x * image_width)
                        y_px = int(landmark.y * image_height)
                        
                        print(f"3D Coordinates: (x: {x}, y: {y}, z: {z}), Pixel Coordinates: (x: {x_px}, y: {y_px})")
                '''
                # Pose ###############################################################
                if enable_segmentation and results.segmentation_mask is not None:
                    # セグメンテーション
                    mask = np.stack((results.segmentation_mask, ) * 3,
                                    axis=-1) > segmentation_score_th
                    bg_resize_image = np.zeros(image.shape, dtype=np.uint8)
                    bg_resize_image[:] = (0, 255, 0)
                    debug_image = np.where(mask, debug_image, bg_resize_image)

                pose_landmarks = results.pose_landmarks
                if pose_landmarks is not None:
                    # 描画
                    debug_image,sholder_z,sholder_x = draw_pose_landmarks(debug_image, pose_landmarks,)


                # ランドマークのインデックス
                LEFT_EYE_INDEX = 467
                RIGHT_EYE_INDEX = 246
                NOSE_INDEX = 1
                CHIN_INDEX = 152# 顎
                if face_landmarks is not None:
                    landmarks=results.face_landmarks.landmark
                    # 3D座標での左目、右目の座標取得
                    left_eye_3d = np.array([landmarks[LEFT_EYE_INDEX].x, landmarks[LEFT_EYE_INDEX].y, landmarks[LEFT_EYE_INDEX].z])
                    right_eye_3d = np.array([landmarks[RIGHT_EYE_INDEX].x, landmarks[RIGHT_EYE_INDEX].y, landmarks[RIGHT_EYE_INDEX].z])
                    # 3D空間での顔の幅の計算
                    face_width_3d = np.linalg.norm(right_eye_3d - left_eye_3d)
                    # 顎の3D座標取得
                    chin_point_3d = np.array([landmarks[CHIN_INDEX].x, landmarks[CHIN_INDEX].y, landmarks[CHIN_INDEX].z])
                    # 顔の高さの計算（顎から鼻までの距離）
                    face_height_3d = np.linalg.norm(chin_point_3d - left_eye_3d)
                    # 顔の対角線の長さの計算（3D）　これを使う
                    face_diagonal_3d = np.sqrt(face_width_3d**2 + face_height_3d**2)
               
                    #print("#######face_diagonal_3d=",face_diagonal_3d)
                    # 3D顔のランドマークモデル
                    # 単位は任意、顔の各ポイントの3D座標 (X, Y, Z)を定義します。
                    face_3d_model_points = np.array([
                        (0.0, 0.0, 0.0),  # 鼻先
                        (0.0, -330.0, -65.0),  # 顎
                        (-225.0, 170.0, -135.0),  # 左目の左端
                        (225.0, 170.0, -135.0),  # 右目の右端
                        (-150.0, -150.0, -125.0),  # 口の左端
                        (150.0, -150.0, -125.0)   # 口の右端
                    ])
                    # 2D顔のランドマーク（顔認識で取得）
                    #face_2d_image_points = [landmark_point[1], # 鼻先
                    #                        landmark_point[152],# 顎
                    #                        landmark_point[246],# 左目の左端
                    #                        landmark_point[466],# 右目の右端
                    #                        landmark_point[78],
                    #                        landmark_point[308],
                    #                        ]
                    nose_tip = (landmark_point[1][0],landmark_point[1][1])
                    # すべてのランドマークを鼻先基準に平行移動
                    face_2d_image_points = [
                        ( int((landmark_point[1][0] - nose_tip[0])/face_diagonal_3d), int((landmark_point[1][1] - nose_tip[1]) /face_diagonal_3d) ),  # 鼻先（原点になる）
                        ( int((landmark_point[152][0] - nose_tip[0])/face_diagonal_3d), int((landmark_point[152][1] - nose_tip[1])/face_diagonal_3d) ),  # 顎
                        ( int((landmark_point[246][0] - nose_tip[0])/face_diagonal_3d), int((landmark_point[246][1] - nose_tip[1])/face_diagonal_3d)),  # 左目の左端
                        ( int((landmark_point[466][0] - nose_tip[0])/face_diagonal_3d), int((landmark_point[466][1] - nose_tip[1])/face_diagonal_3d) ),  # 右目の右端
                        ( int((landmark_point[78][0] - nose_tip[0])/face_diagonal_3d), int((landmark_point[78][1] - nose_tip[1])/face_diagonal_3d) ),   # 左口角
                        ( int((landmark_point[308][0] - nose_tip[0])/face_diagonal_3d), int((landmark_point[308][1] - nose_tip[1])/face_diagonal_3d) )   # 右口角
                    ]
                    face_2d_image_points_array = np.array(face_2d_image_points, dtype=np.float64)   # NumPy配列に変換
                    face_2d_image_points = face_2d_image_points_array.reshape(-1, 2)                # 形状を (N, 2) に整形
                    # カメラ行列（焦点距離、光学中心の定義）
                    image_height, image_width, _ =debug_image.shape  # 画像のサイズ (width, height) image_height, image_width, _ = i
                    size=(image_width,image_height)
                    focal_length = size[1]
                    center = (size[1] / 2, size[0] / 2)
                    camera_matrix = np.array([
                        [focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]
                    ], dtype="double")
                    dist_coeffs = np.zeros((4, 1))                 # レンズ歪み係数 (ここではゼロにしています)
                    # PnP問題を解くことで回転ベクトルと平行移動ベクトルを求める
                    success, rotation_vector, translation_vector = cv.solvePnP(
                        face_3d_model_points, face_2d_image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_EPNP)
                    rotation_matrix, _ = cv.Rodrigues(rotation_vector)                 # 回転ベクトルを回転行列に変換
                    angles, _, _, _, _, _ = cv.RQDecomp3x3(rotation_matrix)            # 回転行列からオイラー角（ピッチ、ヨー、ロール）を計算
                    #距離正規化用のHEADの高さ計算
                    face_size_v=(int(landmark_point[152][1]) - nose_tip[1]) - (int(landmark_point[10][1])-nose_tip[1])
                    face_size_h=(int(landmark_point[152][0]) - nose_tip[0]) - (int(landmark_point[10][0])-nose_tip[0])
                    face_l=math.sqrt(face_size_v**2 + face_size_h**2)#斜めでも正確になるように傾いたときのＸとＹから対角線を計算
                    face_l_r=face_l/250                              #カメラから40cm程度が200なので、ここを基準位置にした
                    print(f"face_size_l_r: {face_l_r:.2f}")
                    # pitch, yaw, rollの計算
                    pitch, yaw, roll = angles  # pitchとroll が正確でないので他の方法で計算、yawは使う
                    eye_slope = (landmark_point[246][1]) - (landmark_point[466][1] )  # 左目の左端-右目の右端
                    f_ud= ((landmark_point[14][1]+landmark_point[13][1])/2 -(landmark_point[152][1])+66)*5
                    roll = eye_slope/face_diagonal_3d # face_diagonal_3d=顔の対角線の長さの計算（3D）
                    pitch= ((-f_ud+120) *face_diagonal_3d)*10
                    #pitch= (-f_ud/(face_diagonal_3d*2))
                    pitch= (pitch/1.3*face_l_r) #距離で正規化
                    
                    print(f"Pitch_c: {pitch:.2f}")#f_ud 　顔の前後傾き
                    print(f"Yaw: {yaw:.2f}")   # 顔の回転
                    print(f"Roll: {roll:.2f}") #eye_slope　顔の左右傾き
                    print(f"眉L: {lb_dist:.2f}") #lb_dist
                    print(f"眉R: {rb_dist:.2f}")#rb_dist,m_out
                    print(f"口: {m_out}")       #m_out
                    print(f"体回転: {sholder_z:.2f}")#sholder_z
                    print(f"体傾き: {sholder_x:.2f}")#,sholder_x 
                    print(f"左目: {left_eye_value:.2f}")#left_eye_value
                    print(f"右目: {right_eye_value:.2f}")#right_eye_value

                    #==============================================================================-
                    # 画像の表示
                    image =debug_image
                    # 矢印の長さのスケール調整（適宜調整してください）
                    arrow_length = 10
                    # 中心点（矢印の始点）
                    center_point = (image.shape[1] // 2, image.shape[0] // 2)
                    # 矢印の色と太さ
                    color_pitch = (255, 0, 0)  # 赤色
                    color_yaw = (0, 255, 0)    # 緑色
                    color_roll = (0, 0, 255)   # 青色
                    thickness = 2
                    # Pitch（上下）の矢印の方向（Y軸）
                    pitch_end_point = (
                        center_point[0], 
                        center_point[1] - int(pitch * arrow_length/10)  # 上下方向に矢印
                    )
                    # Yaw（左右）の矢印の方向（X軸）
                    yaw_end_point = (
                        center_point[0] + int(yaw * arrow_length/10),  # 左右方向に矢印
                        center_point[1]
                    )
                    # Roll（回転）の矢印（円弧で表現）
                    roll_radius = int(arrow_length*10/ 2)
                    roll_start_angle = -45  # 度単位（適宜調整）
                    roll_end_angle = roll_start_angle + int(roll * 10)  # 度単位（適宜調整）
                    # Pitchの矢印を描画
                    cv.arrowedLine(image, center_point, pitch_end_point, color_pitch, thickness, tipLength=0.3)
                    # Yawの矢印を描画
                    cv.arrowedLine(image, center_point, yaw_end_point, color_yaw, thickness, tipLength=0.3)
                    # Rollを円弧で描画
                    cv.ellipse(image, center_point, (roll_radius, roll_radius), 0, roll_start_angle, roll_end_angle, color_roll, thickness)
                    # 矢印と角度の説明用テキスト
                    font = cv.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    cv.putText(image, f"Pitch: {pitch:.2f}", (10,60), font, font_scale, color_pitch, thickness, cv.LINE_AA)
                    cv.putText(image, f"Yaw: {yaw:.2f}", (10, 90), font, font_scale, color_yaw, thickness, cv.LINE_AA)
                    cv.putText(image, f"Roll: {roll:.2f}", (10,120), font, font_scale, color_roll, thickness, cv.LINE_AA)
                    #==============================================================================-
                    # FPS表示
                    fps_color = (0, 255, 0)
                    cv.putText(image, "FPS:" + str(display_fps), (10, 30),
                               cv.FONT_HERSHEY_SIMPLEX, 1.0, fps_color, 2, cv.LINE_AA)            
                    # キー処理(ESC：終了) #################################################
                    key = cv.waitKey(1)
                    if key == 27:  # ESC
                        break
                    # 画面反映 #############################################################
                    cv.imshow('MediaPipe Holistic Demo',image)
            else:
                cv.destroyAllWindows()
                sleep(0.1)
        # キー処理(ESC：終了) ###
        cap.release()
        cv.destroyAllWindows()

def draw_face_landmarks(image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        for index, landmark in enumerate(landmarks.landmark):
            if landmark.visibility < 0 or landmark.presence < 0:
                continue
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_z = landmark.z
            landmark_point.append((landmark_x, landmark_y))
        cv.circle(image, landmark_point[152], 5, (0, 255, 0), 1)    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<, ここ顎です。
        if len(landmark_point) > 0:
            # 参考：https://github.com/tensorflow/tfjs-models/blob/master/facemesh/mesh_map.jpg
            starttime=time.time()
            # 左眉毛(55：内側、46：外側)
            cv.line(image, landmark_point[55], landmark_point[65], (255,0, 0),2)
            cv.line(image, landmark_point[65], landmark_point[52], (255,0, 0),2)
            cv.line(image, landmark_point[52], landmark_point[53], (0, 255, 0), 2)
            cv.line(image, landmark_point[53], landmark_point[46], (0, 255, 0), 2)

            # 右眉毛(285：内側、276：外側)
            cv.line(image, landmark_point[285], landmark_point[295], (0,0, 255),2)
            cv.line(image, landmark_point[295], landmark_point[282], (0,0, 255),2)
            cv.line(image, landmark_point[282], landmark_point[283], (0, 255, 0),2)
            cv.line(image, landmark_point[283], landmark_point[276], (0, 255, 0),2)

            # 左目 (133：目頭、246：目尻)
            cv.line(image, landmark_point[133], landmark_point[173], (255,0, 0),2)
            cv.line(image, landmark_point[173], landmark_point[157], (255,0, 0),2)
            cv.line(image, landmark_point[157], landmark_point[158], (255,0, 0),2)
            cv.line(image, landmark_point[158], landmark_point[159], (255,0, 0),2)
            cv.line(image, landmark_point[159], landmark_point[160], (0, 255, 0),2)
            cv.line(image, landmark_point[160], landmark_point[161], (0, 255, 0),2)
            cv.line(image, landmark_point[161], landmark_point[246], (0, 255, 0),2)

            cv.line(image, landmark_point[246], landmark_point[163], (0, 255, 0),2)
            cv.line(image, landmark_point[163], landmark_point[144], (0, 255, 0),2)
            cv.line(image, landmark_point[144], landmark_point[145], (0, 255, 0),2)
            cv.line(image, landmark_point[145], landmark_point[153], (0, 255, 0),2)
            cv.line(image, landmark_point[153], landmark_point[154], (0, 255, 0),2)
            cv.line(image, landmark_point[154], landmark_point[155], (0, 255, 0),2)
            cv.line(image, landmark_point[155], landmark_point[133], (0, 255, 0),2)
            #*****************左目の開き具合***************************************************
            cv.line(image, landmark_point[159], landmark_point[145], (0,255,255), 2)
            le_top_bottom = landmark_point[145][1]-landmark_point[159][1]
            le_left_right = landmark_point[133][0]-landmark_point[246][0]
            left_eye_value = 1.0-((le_top_bottom/le_left_right)*10)/4
            #print("left-eye:",le_left_right,le_top_bottom, left_eye_value)
            #*****************
            cv.line(image, landmark_point[52], landmark_point[159],(0,255,255), 2)
            lb_dist = landmark_point[52][1]-landmark_point[159][1]

            # 右目 (362：目頭、466：目尻)
            cv.line(image, landmark_point[362], landmark_point[398], (0,0, 255),2)
            cv.line(image, landmark_point[398], landmark_point[384], (0,0, 255),2)
            cv.line(image, landmark_point[384], landmark_point[385], (0,0, 255),2)
            cv.line(image, landmark_point[385], landmark_point[386], (0,0, 255),2)
            cv.line(image, landmark_point[386], landmark_point[387], (0, 255, 0),2)
            cv.line(image, landmark_point[387], landmark_point[388], (0, 255, 0),2)
            cv.line(image, landmark_point[388], landmark_point[466], (0, 255, 0),2)

            cv.line(image, landmark_point[466], landmark_point[390], (0, 255, 0),2)
            cv.line(image, landmark_point[390], landmark_point[373], (0, 255, 0),2)
            cv.line(image, landmark_point[373], landmark_point[374], (0, 255, 0),2)
            cv.line(image, landmark_point[374], landmark_point[380], (0, 255, 0),2)
            cv.line(image, landmark_point[380], landmark_point[381], (0, 255, 0),2)
            cv.line(image, landmark_point[381], landmark_point[382], (0, 255, 0),2)
            cv.line(image, landmark_point[382], landmark_point[362], (0, 255, 0),2)
            #*****************右目の開き具合***************************************************
            cv.line(image, landmark_point[386], landmark_point[374], (0,255,255), 2)
            re_top_bottom = landmark_point[374][1]-landmark_point[386][1]
            re_left_right = landmark_point[466][0]-landmark_point[362][0]
            if re_left_right==0.0:
                re_left_right=0.01
            right_eye_value = 1.0-((re_top_bottom/re_left_right)*10)/4
            #print("right-eye:",re_left_right,re_top_bottom, right_eye_value)
            #*****************
            cv.line(image, landmark_point[282], landmark_point[386],(0,255,255), 2)
            rb_dist = landmark_point[282][1]-landmark_point[386][1]

            # 口 (308：右端、78：左端)
            cv.line(image, landmark_point[308], landmark_point[415], (255,0, 0),2)
            cv.line(image, landmark_point[415], landmark_point[310], (255,0, 0),2)
            cv.line(image, landmark_point[310], landmark_point[311], (255,0, 0),2)
            cv.line(image, landmark_point[311], landmark_point[312], (255,0, 0),2)
            cv.line(image, landmark_point[312], landmark_point[13],(255,0, 0),2)
            cv.line(image, landmark_point[13], landmark_point[82], (0, 255, 0), 2)
            cv.line(image, landmark_point[82], landmark_point[81], (0, 255, 0), 2)
            cv.line(image, landmark_point[81], landmark_point[80], (0, 255, 0), 2)
            cv.line(image, landmark_point[80], landmark_point[191], (0, 255, 0), 2)
            cv.line(image, landmark_point[191], landmark_point[78], (0, 255, 0), 2)

            cv.line(image, landmark_point[78], landmark_point[95], (0,0, 255),2)
            cv.line(image, landmark_point[95], landmark_point[88], (0,0, 255),2)
            cv.line(image, landmark_point[88], landmark_point[178], (0,0, 255),2)
            cv.line(image, landmark_point[178], landmark_point[87],(0,0, 255),2)
            cv.line(image, landmark_point[87], landmark_point[14], (0,0, 255),2)
            cv.line(image, landmark_point[14], landmark_point[317], (0, 255, 0), 2)
            cv.line(image, landmark_point[317], landmark_point[402], (0, 255, 0),2)
            cv.line(image, landmark_point[402], landmark_point[318], (0, 255, 0),2)
            cv.line(image, landmark_point[318], landmark_point[324], (0, 255, 0),2)
            cv.line(image, landmark_point[324], landmark_point[308], (0, 255, 0),2)
            #*****************口の開き具合***************************************************
            cv.line(image, landmark_point[308], landmark_point[74], (0,255,255), 2)
            m_lr_value= landmark_point[308][0]-landmark_point[74][0]
            cv.line(image, landmark_point[14], landmark_point[13], (0,255,255), 2)
            m_td_value= landmark_point[14][1]-landmark_point[13][1]
            print(">>>>>>>>>>>>>>>>>>>>>>>>>",time.time()- starttime)
            if m_lr_value==0.0:
                m_lr_value=0.01
            m_ratio= m_td_value/m_lr_value
            #print("m_ratio=",m_lr_value,m_td_value,m_ratio)
            if m_ratio>0.3:
                m_out="eee"
            elif m_ratio>0.25:
                m_out="ooo"
            elif m_ratio>0.2:
                m_out="aaa"
            elif m_ratio>0.05:
                m_out="uuu"
            elif m_ratio>0.03:
                m_out="iii"
            else:
                m_out=""
            #print(m_out)
        return image,landmark_point,lb_dist,rb_dist,m_out,left_eye_value,right_eye_value

def draw_pose_landmarks(
        image,
        landmarks,
        visibility_th=0.5,
        ):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        for index, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_z = landmark.z
            landmark_point.append([landmark.visibility, (landmark_x, landmark_y),landmark_z])

            if landmark.visibility < visibility_th:
                continue
            if index == 0:  # 鼻
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if len(landmark_point) > 0:
            # 肩
            if landmark_point[11][0] > visibility_th and landmark_point[12][0] > visibility_th:
                cv.line(image, landmark_point[11][1], landmark_point[12][1],(0, 255, 0), 2)
                sholder_z = landmark_point[11][2]-landmark_point[12][2]#肩の回転 -0.8~0.8　だが大きな回転。通常は-0.5〜+0.5
                sholder_x = landmark_point[11][1][1]-landmark_point[12][1][1]#肩の傾き
        return image,sholder_z,sholder_x

if __name__ == "__main__":
    import uvicorn
    '''
    # 引数解析 #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    smooth_landmarks = not args.unuse_smooth_landmarks
    enable_segmentation = args.enable_segmentation
    model_complexity = args.model_complexity
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    segmentation_score_th = args.segmentation_score_th
    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    # モデルロード #############################################################
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
                model_complexity=model_complexity,
                smooth_landmarks=smooth_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence, )
    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    get_mocap(cvFpsCalc,holistic,cap,segmentation_score_th,enable_segmentation)
    '''
    threads_mocap= threading.Thread(target=init) #layer_mix_thを呼び出すスレッド
    threads_mocap.start() #layer_mix_th開始

    uvicorn.run(app, host="127.0.0.1", port=3005)
