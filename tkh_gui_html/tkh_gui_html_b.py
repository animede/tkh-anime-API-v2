import httpx
from fastapi import FastAPI, File, Form, UploadFile, WebSocket,WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse,Response
from fastapi.staticfiles import StaticFiles
from concurrent.futures import ThreadPoolExecutor
import asyncio
from pydantic import BaseModel
from typing import Optional
import base64
import numpy as np
from PIL import Image
import io
import cv2
import time
from time import sleep
import signal
import sys
import json
from poser_client_tkhmp_upmp_v1_3b_class import TalkingHeadAnimefaceInterface
from collections import deque

global result_out_image
global img_mode
global img_number
global current_pose_dic
global exec_flag
global mocap_mode
global current_image


app = FastAPI()

#外部サーバ
#tkh_url = 'http://192.168.5.89:8002'
#esr_url = 'http://192.168.5.89:8018/resr_upscal/'

#ローカルサーバ
tkh_url = 'http://127.0.0.1:8001'
esr_url = 'http://127.0.0.1:8008/resr_upscal/'

GET_MCAP_URL = "http://localhost:3005/get_mcap/"

app.mount("/static", StaticFiles(directory="static"), name="static")
Thi = TalkingHeadAnimefaceInterface(tkh_url)
pose_dic_org = Thi.get_init_dic()
pose_dic=pose_dic_org.copy() #Pose 初期値
current_pose_dic=pose_dic.copy()
img_number = 0
user_id=0

#フレーム時間
f_time=0.07
#アップスケールとtkhプロセスの開始
Thi.create_mp_upscale(esr_url)
Thi.create_mp_tkh()
# スレッドプールエグゼキュータの設定
executor = ThreadPoolExecutor()

#グローバルフラグ
useOpenCV="off"
exec_flag=False
mocap_mode="stop"

connected_websockets = []  # 接続されたWebSocketを管理するリスト

# 各変数の過去5個の値を保存するデータ構造（キュー）
pitch_history = deque(maxlen=10)
yaw_history = deque(maxlen=10)
roll_history = deque(maxlen=10)
lb_dist_history = deque(maxlen=10)
rb_dist_history = deque(maxlen=10)
sholder_z_history = deque(maxlen=10)
sholder_x_history = deque(maxlen=10)
left_eye_value_history = deque(maxlen=3)
right_eye_value_history = deque(maxlen=3)

pitch_history_acr_pre= 0.0
yaw_history_avr_pre= 0.0
roll_history_avr_pre= 0.0
lb_dist_history_avr_pre= 0.0
rb_dist_history_avr_pre= 0.0
sholder_z_history_avr_pre= 0.0
sholder_x_history_avr_pre= 0.0
left_eye_value_history_avr_pre= 0.0
right_eye_value_history_avr_pre= 0.0

def signal_handler(signal, frame):
    print("Ctrl-C pressed: Exiting...")
    Thi.up_scale_proc_terminate()
    Thi.tkh_proc_terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)  

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open('static/index.html', 'r') as f:
        return f.read()


@app.post("/get-image/")
async def get_image():
    global current_image

    # OpenCVイメージをJPEG形式にエンコード
    _, image_encoded = cv2.imencode('.png', current_image)

    # バイトデータに変換
    image_bytes = image_encoded.tobytes()

    # Responseでバイナリデータを返送
    return Response(content=image_bytes, media_type="application/octet-stream")
    

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    global result_out_image
    global img_number
    contents = await file.read()
    input_image = Image.open(io.BytesIO(contents))
    result_image=Thi.image_2_form(input_image, "pil")
    cv2_image = np.array(result_image, dtype=np.uint8)
    result_out_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGBA2BGRA)
    img_number = Thi.load_img(result_image, user_id)

    buffered = io.BytesIO()
    result_image.save(buffered, format="PNG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return {"original": base64.b64encode(contents).decode('utf-8'), "processed": encoded_string,"img_number":img_number}

@app.post("/generate_image/")
def generate_image(mode: str = Form(...), scale: int = Form(...), fps: int = Form(...)):
    global result_out_image
    global img_mode
    global img_number
    global current_pose_dic

    print("+++++++++++++++++++++++ mode=",mode,scale,fps)
    try:
        cv2_image = np.array(result_image, dtype=np.uint8)
        result_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
    except:
        print("image=cv2")
    if len(mode)>9:  # <= modeがクロップ用の位置情報リストで来た場合。すべての要素が1桁の場合やmodeの文字が10を超えるとだめです
        img_mode = json.loads(mode)
    else:
        img_mode=mode
    #user_id=0
    fps=0 #no waite
    gen_image,result = Thi.mp_dic2image_frame(result_out_image,current_pose_dic,img_number,user_id,img_mode,scale,fps,up=False)
    if result == False:
        while result == False:
            gen_image, result =  Thi.mp_get_inference_queu()
    if result == True:
        _,result=Thi.mp_get_upscale(result_out_image,gen_image,img_mode,scale,fps)
    return {"status": "success"}

# 調整データ用のモデル
'''
class AdjustmentData(BaseModel):
    eyebrow_type: str
    eye_type: str
    mouth_type: str
    adjustment_type: str
    adjustment_value: float
    scale: str
    fps: str
    useOpenCV: str
@app.post("/update_adjustment/")
def update_adjustment(data: AdjustmentData):
'''
@app.post("/update_adjustment/")
def update_adjustment(eyebrow_type: str = Form(...), eye_type: str = Form(...),  mouth_type: str = Form(...),adjustment_type: str = Form(...), adjustment_value: str = Form(...), scale: str = Form(...), fps:str = Form(...),useOpenCV2:str = Form(...)):
    global result_out_image
    global img_mode
    global img_number
    global current_pose_dic
    global useOpenCV
    '''
    eyebrow_type = data.eyebrow_type
    eye_type = data.eye_type
    mouth_type = data.mouth_type
    adjustment_type = data.adjustment_type
    adjustment_value = data.adjustment_value  # floatに変換不要（すでにfloat）
    scale = int(data.scale)  # scaleを整数に変換
    fps=int(data.fps)
    useOpenCV = data.useOpenCV  # useOpenCVは文字列として取得
    '''
    adjustment_value=float(adjustment_value)
    scale = int(scale)  # scaleを整数に変換
    useOpenCV = useOpenCV2
    while True:
            # adjustment_typeを確認して対応するキーを更新
            if adjustment_type=="eyebrow" or adjustment_type=="eye" or adjustment_type=="iris_small":
                current_pose_dic["eyebrow"]["menue"]=eyebrow_type
                current_pose_dic["eye"]["menue"]=eye_type
                current_pose_dic[adjustment_type]["left"]=adjustment_value
                current_pose_dic[adjustment_type]["right"]=adjustment_value
            elif adjustment_type=="iris_rotation":
                current_pose_dic["iris_rotation"]["x"]=adjustment_value
                current_pose_dic["iris_rotation"]["y"]=adjustment_value
            elif adjustment_type=="mouth":
                current_pose_dic["mouth"]["menue"]=mouth_type
                current_pose_dic["mouth"]["val"]=adjustment_value
            elif adjustment_type=="neck":
                current_pose_dic["neck"]=-adjustment_value*2.0 #首の最大は2.0
            else: # 'head_x' と 'head_y'
                part, axis = adjustment_type.split("_")
                current_pose_dic[part][axis] = -adjustment_value*2.0 #頭と体の最大は2.0
            print(current_pose_dic)

            fps=0 #no waite
            gen_image,result = Thi.mp_dic2image_frame(result_out_image,current_pose_dic,img_number,user_id,img_mode,scale,fps,up=False)
            if result == False:
                while result == False:
                    gen_image, result =  Thi.mp_get_inference_queu()
            if result == True:
                _,result=Thi.mp_get_upscale(result_out_image,gen_image,img_mode,scale,fps)
            return {"status": "success"}

# WebSocketで定期的に画像を送信
@app.websocket("/ws/update-image/")
async def websocket_update_image(websocket: WebSocket):
    global result_out_image
    global current_image
    
    await websocket.accept()
    connected_websockets.append(websocket)  # 接続されたWebSocketをリストに追加
    try:
        while True:
            result_out_image, result =  Thi.mp_get_upscale_queu()
            # 結果が準備できている場合のみ送信
            if result == True:
                current_image=result_out_image
                try:
                    if useOpenCV=="On":
                        cv2.imshow("Loaded image",result_out_image)
                        cv2.waitKey(1)
                    else:
                        try:
                            cv2.destroyWindow("Loaded image")
                        except:
                            err="Loaded image is not exist"
                    # OpenCVの画像からPILに変換
                    cv2_image = cv2.cvtColor(result_out_image, cv2.COLOR_BGRA2RGBA)
                    out_image = Image.fromarray(cv2_image)
                    # 画像をバイナリデータに変換
                    buffered = io.BytesIO()
                    out_image.save(buffered, format="PNG")
                    buffered.seek(0)
                    image_data = buffered.read()
                    await websocket.send_bytes(image_data) # WebSocketでバイナリデータを送信
                    print("Image sent via WebSocket")
                except Exception as e:
                    print(f"Error during image conversion or sending: {e}")
            await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
        websocket.remove(websocket)
    except Exception as e:
        print(f"Error occurred: {e}")
        websocket.remove(websocket)

@app.post("/process-emotions/")
def process_emotions(emotions: str = Form(...), mode: Optional[str] = Form(None), scale: Optional[int] = Form(None), fps: Optional[int] = Form(None),useOpenCV2:Optional[str] = Form(None),intensity:Optional[str] = Form(None)):
    global result_out_image
    global img_mode 
    global img_number
    global current_pose_dic
    from emotion import emotion
    global useOpenCV

    useOpenCV=useOpenCV2
    intensity=float(intensity)
    # ここでemotions_listと他のフォームデータを使用した処理を実装
    print("+++++Value=",emotions,mode,scale,fps,useOpenCV,intensity)
    if emotions=="init":#初期化
        current_pose_dic=Thi.get_init_dic()
    current_pose_dic=emotion(current_pose_dic,emotions,intensity)
    fps=0 #no waite
    gen_image,result = Thi.mp_dic2image_frame(result_out_image,current_pose_dic,img_number,user_id,img_mode,scale,fps,up=False)
    if result == False:
        while result == False:
            gen_image, result =  Thi.mp_get_inference_queu()
    if result == True:
        _,result=Thi.mp_get_upscale(result_out_image,gen_image,img_mode,scale,fps)
    return {"status": "success"}

#923 tkh系のクラスを使う他のエンドポイントもasyn対応にすることで動くようになりました
@app.post("/auto-process/")
async def auto_process(test: Optional[str] = Form(None),mode: Optional[str] = Form(None), scale: Optional[int] = Form(None),fps: Optional[int] = Form(None),useOpenCV:Optional[str] = Form(None)):
    global result_out_image
    global img_mode 
    global img_number
    global current_pose_dic
    global exec_flag

    from poser_generater_v1_3_autopose_test import auto_pose_1
    #print("--->",test,mode,scale,fps)

    if exec_flag==True:
        return {"status": "success"}
    exec_flag=True
    pose_dic_org=Thi.get_init_dic()
    auto_pose_1(Thi,pose_dic_org,test,result_out_image,user_id,img_number,img_mode ,scale,fps)
    websocket = connected_websockets[0]
    for result_out_image in auto_pose_1(Thi, pose_dic_org, test, result_out_image, user_id, img_number, img_mode, scale, fps):
        #try:
            if useOpenCV=="On":
                cv2.imshow("Loaded image",result_out_image)
                cv2.waitKey(1)
            else:
                try:
                    cv2.destroyWindow("Loaded image")
                except:
                    err="Loaded image is not exist"
            # OpenCVの画像からPILに変換
            cv2_image = cv2.cvtColor(result_out_image, cv2.COLOR_BGRA2RGBA)
            out_image = Image.fromarray(cv2_image)
            # 画像をバイナリデータに変換
            buffered = io.BytesIO()
            out_image.save(buffered, format="PNG")
            buffered.seek(0)
            image_data = buffered.read()
            await websocket.send_bytes(image_data) # WebSocketでバイナリデータを送信
    try:
        cv2.destroyWindow("Loaded image")
    except:
        print="Loaded image is not exist"
    exec_flag=False
    return {"status": "success"}
    
async def rprocess_term():
    #サブプロセスの終了
    Thi.up_scale_proc_terminate()
    Thi.tkh_proc_terminate()

def update_and_get_moving_average(value, history):
    history.append(value)     # 新しい値を履歴に追加
    return sum(history) / len(history)    # 履歴の平均を返す

# 非同期でスレッド処理する関数
def fetch_mcap_data(mode: str,scale: int,fps: int,useOpenCV2:Optional[str]):
    global result_out_image
    global img_mode
    global img_number
    global current_pose_dic
    global useOpenCV
    global mocap_mode

    response_data = {}
    #try:
    while mocap_mode=="start":
            start_time = time.time()
            # 非同期クライアントでリクエストを送信
            with httpx.Client() as client:
                response = client.post(GET_MCAP_URL, data={"mode": mode})
                response_data = response.json()
            # 変数にレスポンスデータを格納
            pitch = response_data["pitch"] # 顔の前後傾き
            yaw = response_data["yaw"]     # 顔の回転
            roll = response_data["roll"]   # 顔の左右傾き
            lb_dist = response_data["lb_dist"]
            rb_dist = response_data["rb_dist"]
            m_out = response_data["m_out"]
            #sholder_z = response_data["sholder_z"]
            #sholder_x = response_data["sholder_x"]
            sholder_x = -response_data["sholder_z"]*10
            sholder_z = -response_data["sholder_x"]/10
            left_eye_value = response_data["left_eye_value"]
            right_eye_value = response_data["right_eye_value"]
            # 各数値の異常値を無視　目は対象外（瞬きするから）
            if len(pitch_history)>10:
                if (abs(pitch_history_acr_pre*0.3)<=abs(pitch)<=abs(pitch_history_acr_pre*1.3))!=True:
                    pitch= pitch_history_acr_pre
                if (abs(yaw_history_avr_pre*0.3)<=abs(yaw)<=abs(yaw_history_avr_pre*1.3))!=True:
                    yaw= yaw_history_avr_pre
                if (abs(roll_history_avr_pre*0.3)<=abs(roll)<=abs(roll_history_avr_pre*1.3))!=True:
                    roll= roll_history_avr_pre
                if (abs(lb_dist_history_avr_pre*0.3)<=abs(lb_dist)<=abs(lb_dist_history_avr_pre*1.3))!=True:
                    lb_dist= lb_dist_history_avr_pre
                if (abs(rb_dist_history_avr_pre*0.3)<=abs(rb_dist)<=abs(rb_dist_history_avr_pre*1.3))!=True:
                    rb_dist= rb_dist_history_avr_pre
                if (abs(sholder_z_history_avr_pre*0.3)<=abs(sholder_z)<=abs(sholder_z_history_avr_pre*1.3))!=True:
                    sholder_z= sholder_z_history_avr_pre
                if (abs(sholder_x_history_avr_pre*0.3)<=abs(sholder_x)<=abs(sholder_x_history_avr_pre*1.3))!=True:
                    sholder_x= sholder_x_history_avr_pre
            # 各数値の移動平均を計算
            pitch_avg = update_and_get_moving_average(pitch, pitch_history)
            yaw_avg = update_and_get_moving_average(yaw, yaw_history)
            roll_avg = update_and_get_moving_average(roll, roll_history)
            lb_dist_avg = update_and_get_moving_average(lb_dist, lb_dist_history)
            rb_dist_avg = update_and_get_moving_average(rb_dist, rb_dist_history)
            sholder_z_avg = update_and_get_moving_average(sholder_z, sholder_z_history)
            sholder_x_avg = update_and_get_moving_average(sholder_x, sholder_x_history)
            left_eye_value_avg = update_and_get_moving_average(left_eye_value, left_eye_value_history)
            right_eye_value_avg = update_and_get_moving_average(right_eye_value, right_eye_value_history)
            # 異常値が出現したときに古い平均を使えるよう記憶
            pitch_history_acr_pre=  pitch_avg                    
            yaw_history_avr_pre= yaw_avg
            roll_history_avr_pre= roll_avg
            lb_dist_history_avr_pre= lb_dist_avg
            rb_dist_history_avr_pre= rb_dist_avg
            sholder_z_history_avr_pre= sholder_z_avg
            sholder_x_history_avr_pre= sholder_x_avg
            print(f"Pitch: {pitch}, Yaw: {yaw:.2f}, Roll: {roll},体回転:{sholder_z:.2f},体傾き:{sholder_x:.2f}")
            print(f"左目:{left_eye_value:.2f}, 右目:{right_eye_value:.2f},眉L:{lb_dist:.2f},眉R:{rb_dist:.2f},口: {m_out}")
            # ここで変数に対してHKHに合うよう数値を合わせる処理を行い、current_pose_dicをアップデート
            current_pose_dic["eye"]["left"]=(left_eye_value_avg-0.1)*2
            current_pose_dic["eye"]["right"]=(right_eye_value_avg-0.1)*2
            current_pose_dic["head"]["x"]=pitch_avg
            current_pose_dic["head"]["y"]=yaw_avg
            current_pose_dic["neck"]=roll_avg*2
            current_pose_dic["body"]["z"]=sholder_z_avg*2
            current_pose_dic["body"]["y"]=sholder_x_avg*2
            current_pose_dic["eyebrow"]["left"]=lb_dist_avg
            current_pose_dic["eyebrow"]["right"]=rb_dist_avg
            if m_out=="": 
                m_out="iii" # 口の値がない場合は形状”い”で最小指定して口を閉じる
                current_pose_dic["mouth"]["menue"]=m_out
                current_pose_dic["mouth"]["val"]=0.0
            else:           # 口の値がある場合はm_outに従いcurrent_pose_dic["mouth"]["menue"]を設定し、最大値を指定
                current_pose_dic["mouth"]["menue"]=m_out
                current_pose_dic["mouth"]["val"]=1.0
            # current_pose_dicをTkhに渡して画像を生成
            fps=0 #no waite 指定
            gen_image,result = Thi.mp_dic2image_frame(result_out_image,current_pose_dic,img_number,user_id,img_mode,scale,fps,up=False)
            if result == False:
                while result == False:  #出力が生成されるまで、待つ
                    gen_image, result =  Thi.mp_get_inference_queu()
            if result == True:          #生成された画像をアップスケーラに渡す。アップスケール後の画像はdef auto_process()でQueuをチェックして受取る
                _,result=Thi.mp_get_upscale(result_out_image,gen_image,img_mode,scale,fps)
            # 過剰なTKHへの生成リクエストをしないよう、指定したフレームレートになるまでsleepする
            sleeptime=f_time-(time.time()-start_time) 
            if(sleeptime>0):
                sleep(sleeptime)

@app.post("/mcap/")
async def mcap(mode: str = Form(...), scale: int = Form(...), fps: int = Form(...),useOpenCV2:Optional[str] = Form(None)):
    global mocap_mode
    global useOpenCV

    useOpenCV=useOpenCV2
    print("mode=",mode)
    mocap_mode=mode
    async with httpx.AsyncClient() as client:
        response = await client.post(GET_MCAP_URL, data={"mode": mode})
    # スレッドでfetch_mcap_dataを実行
    loop = asyncio.get_running_loop()
    loop.run_in_executor(executor, fetch_mcap_data, mode,scale,fps,useOpenCV2)
    #return mcap_data # 返したい場合は必要に応じてレスポンスを返す

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3001)

