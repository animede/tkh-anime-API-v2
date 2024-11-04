事前にCUDA-Toolkitをインストールしてください

https://developer.nvidia.com/cuda-downloads
環境を選ぶとインストールコマンドが表示されるのでそのままコピペします。networkインストールが楽だと思います

パスを通しておきましょう（よく忘れます）
export PATH=/usr/local/cuda:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
source ~/.bashrc

以下でCUDA確認
nvcc -V　
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Wed_Nov_22_10:17:15_PST_2023
Cuda compilation tools, release 12.3, V12.3.107
Build cuda_12.3.r12.3/compiler.33567101_0



リポジトリをクローン
git clone https://github.com/pkhungurn/tkh-anime-API-v2.git

仮想環境作成
python3 -m venv tkh
source tkh/bin/activate
cd tkh-anime-API-v2

pip install -r requirements.txt
 

インストール
pip install requirements.txt

モデルダウンロード
wget https://www.dropbox.com/s/y7b8jl4n2euv8xe/talking-head-anime-3-models.zip?dl=0
data/models/ にtalking-head-anime-3-models.zip?dl=0をコピーして展開

作成されたホルダー名をmodelsに変更

HuggingFaceから
①AnimeFaceDetection
②real-ESRGAN
③AnimeSegmentation
のウエイトをダウンロード
ssd_best8.pthをweightsホルダへコピー
realesr-animevideov3.pthをweightsホルダへコピー
isnetis.ckptはtkh-anime-API-v2のルート（このファイルがあるディレクトリ）へコピー



動かし方

はじめにalking-Head-Animeface-3サーバを動かしておく
***************Talking-Head-Animeface-3   host:127.0.0.1 port:8001
source tkh/bin/activate
cd tkh-anime-API-v2
python poser_api_v1_3S_server.py       <<<<<<<< 11/4日修正

**********************テスト
>>>>>>>>> ベース Talking-head
source tkh/bin/activate
cd tkh-anime-API-v2
python poser_client_v1_3_upscale_test.py --test 7

>>>>>>>>> TKH & アップスケール
python poser_generater_v1_3_test.py --filename kitchen_anime.png


poser_generater_v1_3_autopose_test.py


>>>>>>>>> TKH & アップスケールmp クライアント
python poser_client_tkhmp_upmp_v1_3_upscale_test.py --test 1 --fps 20
python poser_client_tkhmp_upmp_v1_3_upscale_test.py --test 2

注）--test 2はfps=40、scale=4, クリップエリア[55,155,200,202] #[top,left,hight,whith]で強制指定
--filename','-i', default='000002.png', type=str)
--test', default=0, type=int)
--host', default='http://127.0.0.1:8001', type=str)#Talking-Head-Anime3 サーバアドレス
--esr', default='http://127.0.0.1:8008', type=str) #upscale　サーバアドレス
--mode', default='fullbody', type=str) #mode="face", "breastup" , "waistup" , "upperbody" ,"fullbody", "full"
--fps', default=20, type=int) #10~50
--scale', default=2, type=float) #1,2,4
    
    
>>>>>>>>> GUI　test
source tkh/bin/activate
cd tkh-anime-API-v2
python tkh_basic_poser_gui.py

**********************テンプレート作成 テスト
source tkh/bin/activate
cd tkh-anime-API-v2
python poser_image_2_template_class_test.py

**********************モーションキャプチャWebアプリ
>>>>>>>>> mediapipサーバ起動
source tkh/bin/activate
cd tkh-anime-API-v2/mediapipe/
python sample_holistic_server.py

>>>>>>>>> アプリサーバ起動
source tkh/bin/activate             <<<<<<<< 11/4日修正
cd tkh-anime-API-v2/tkh_gui_html    <<<<<<<< 11/4日修正
python tkh_gui_html_b.py

関連URL　以下のリポジトリを参考に開発しています。
https://github.com/pkhungurn/talking-head-anime-3-demo
https://github.com/SkyTNT/anime-segmentation
https://github.com/xinntao/Real-ESRGAN
https://github.com/animede/anime_face_detection
https://ai.google.dev/edge/mediapipe/solutions/guide?hl=ja
http://cedro3.com/ai/mediapipe/
https://github.com/cedro3/mediapipe



