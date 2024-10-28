import numpy as np
import pickle
import cv2
from PIL import Image
import argparse
from time import sleep
import time
from tkh_up_scale_b import upscale
from poser_image_2_template_class import Image2form

url="http://0.0.0.0:8008/resr_upscal/"

#PIL形式の画像を動画として表示
def image_show(imge):
    imge = np.array(imge)
    imge = cv2.cvtColor(imge, cv2.COLOR_RGBA2BGRA)
    cv2.imshow("Loaded image",imge)
    cv2.waitKey(1)

def test1(Thi,input_image):  #inference()のテスト
        current_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0,
                        0.767, 0.566, 0.626,
                        0.747, 0.485,
                        0.444, 0.232,
                        0.646, 1.0]
        
        img_number=Thi.load_img(input_image,user_id=0)
        _,out_image=Thi.inference(input_image,current_pose,"pil")
        return out_image
            
#サンプル ２　inference()を使用　パック形式をリポジトリの形式に変換 　イメージは毎回ロード  #packed_pose=>current_pose2
def test2(Thi,input_image):
        user_id=0
        packed_pose=["happy", [0.5,0.0], "wink", [1.0,0.0], [0.0,0.0], [0.0,0.0], "ooo", [0.0,0.0], [0.0,0.0], 0.0, 0.0,0.0, 0.0]
        current_pose2=Thi.get_pose(packed_pose) #packed_pose=>current_pose2
        img_number=Thi.load_img(input_image,user_id)
        _,out_image=Thi.inference(input_image,current_pose2,"pil")#crop=を指定しない場合はクロップ市内（crop=”non")
        return out_image
    
#サンプル ３ inference(）を使用　Dict形式をget_pose_dicdでリポジトリの形式に変換　 イメージは毎回ロード
def test3(Thi,input_image):
        user_id=0
        #サンプル Dict形式 
        #"mouth"には2種類の記述方法がある"lowered_corner"と”raised_corner”は左右がある
        #  "mouth":{"menue":"aaa","val":0.0},
        #  "mouth":{"menue":"lowered_corner","left":0.5,"right":0.0},　これはほとんど効果がない
        pose_dic={"eyebrow":{"menue":"happy","left":1.0,"right":0.0},
                "eye":{"menue":"wink","left":0.5,"right":0.0},
                "iris_small":{"left":0.0,"right":0.0},
                "iris_rotation":{"x":0.0,"y":0.0},
                "mouth":{"menue":"aaa","val":0.7},
                "head":{"x":0.0,"y":0.0},
                "neck":0.0,
                "body":{"y":0.0,"z":0.0},
                "breathing":0.0
                }
        pose=Thi.get_pose_dic(pose_dic)#Dic-> pose変換
        img_number=Thi.load_img(input_image,user_id)
        _,out_image=Thi.inference(input_image,pose,"pil")#crop=を指定しない場合はクロップ市内（crop=”non")
        return out_image
    
#サンプル ４　inference_pos()を使用　パック形式　イメージは事前ロード
def test4(Thi,img_number):
        user_id=0
        packed_pose=["happy", [0.5,0.0], "wink", [1.0,0.0], [0.0,0.0], [0.0,0.0], "ooo", [0.0,0.0], [0.0,0.0], 0.0, 0.0,0.0, 0.0]
        _,out_image=Thi.inference_pos(packed_pose,img_number,user_id,"cv2")
        out_image = cv2.cvtColor(out_image, cv2.COLOR_BGRA2RGBA)
        return out_image
        
#サンプル 5　inference_dic()を使用 　DICT形式で直接サーバを呼ぶ　イメージは事前ロード    
def test5(Thi,img_number):
        user_id=0
        pose_dic=pose_dic #Pose 初期値
        current_pose_list=[]
        for i in range(20):
            current_pose_dic=pose_dic
            current_pose_dic["eye"]["menue"]="wink"#pose_dicに対して動かしたい必要な部分だけ操作できる
            current_pose_dic["eye"]["left"]=i*2/40 #
            _,out_image = Thi.inference_dic(current_pose_dic,img_number,user_id,"cv2")
            out_image = cv2.cvtColor(out_image, cv2.COLOR_BGRA2RGBA)
            yield out_image
            
 #サンプル 6 inference_img() pose=リポジトリの形式(ベタ書き)　
def test6(Thi,img_number):
        user_id=0
        for i in range(100):
            current_pose3= [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0, 0.0, i/100, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.78, 0.57, 0.63, 0.75, 0.49, 0.43,0.23, 0.65,1.0]
            _,out_image=Thi.inference_img(current_pose3,img_number,user_id,"cv2")
            out_image = cv2.cvtColor(out_image, cv2.COLOR_BGRA2RGBA)
            yield out_image
                   
#サンプル 7　inference_img() poseはパック形式形式をリポジトリの形式に変換 イメージは事前ロード,パック形式で連続変化させる  
def test7(Thi,img_number):
        user_id=0
        for i in range(50):
            packed_current_pose=[
                "happy",[0.5,0.0],"wink", [i/50,0.0], [0.0,0.0], [0.0,0.0],"ooo", [0.0,0.0], [0.0,i*3/50],i*3/50, 0.0, 0.0, 0.0]
            current_pose=Thi.get_pose(packed_current_pose) #packed_pose=>current_pose2
            _,out_image=Thi.inference_img(current_pose,img_number,user_id,"cv2")
            out_image = cv2.cvtColor(out_image, cv2.COLOR_BGRA2RGBA)
            yield out_image
            
        for i in range(100):
            packed_current_pose=[
                "happy", [0.5,0.0], "wink",[1-i/50,0.0], [0.0,0.0], [0.0,0.0], "ooo", [0.0,0.0], [0.0,3-i*3/50], 3-i*3/50, 0.0, 0.0, 0.0,]
            current_pose2=Thi.get_pose(packed_current_pose)#packed_current_pose==>current_pose2
            _,out_image=Thi.inference_img(current_pose2,img_number,user_id,"cv2")
            out_image = cv2.cvtColor(out_image, cv2.COLOR_BGRA2RGBA)
            yield out_image
            
        for i in range(100):
            packed_current_pose=[
                "happy", [0.5,0.0], "wink", [i/100,i/100], [0.0,0.0], [0.0,0.0], "ooo", [0.0,0.0], [0.0,-3+i*3/50], -3+i*3/50,0.0, 0.0,0.0,]
            current_pose2=Thi.get_pose(packed_current_pose)#packed_current_pose==>current_pose2
            _,out_image=Thi.inference_img(current_pose2,img_number,user_id,"cv2")
            out_image = cv2.cvtColor(out_image, cv2.COLOR_BGRA2RGBA)
            yield out_image
            
        for i in range(100):
            packed_current_pose=[
                "happy", [0.5,0.0], "wink", [0.0,0.0], [0.0,0.0], [0.0,0.0], "ooo",  [0.0,0.0], [0.0,3-i*3/100],  3-i*3/100,  0.0, 0.0, 0.0,]
            current_pose2=Thi.get_pose(packed_current_pose) #packed_current_pose==>current_pose2
            _,out_image=Thi.inference_img(current_pose2,img_number,user_id,"cv2")
            out_image = cv2.cvtColor(out_image, cv2.COLOR_BGRA2RGBA)
            yield out_image
        

#サンプル 8　inference_dic() 　poseはDICT形式で直接サーバを呼ぶ　イメージは事前ロード　  DICT形式で必要な部分のみ選んで連続変化させる
def test8(Thi,img_number,pose_dic):
        user_id=0
        div_count=30
        pose_dic=pose_dic #Pose 初期値
        current_pose_list=[]
        for i in range(int(div_count/2)):
            current_pose_dic=pose_dic
            current_pose_dic["eye"]["menue"]="wink"
            current_pose_dic["eye"]["left"]=i/(div_count/2)
            current_pose_dic["head"]["y"]=i*3/(div_count/2)
            current_pose_dic["neck"]=i*3/(div_count/2)
            current_pose_dic["body"]["y"]=i*5/(div_count/2)
            _,out_image = Thi.inference_dic(current_pose_dic,img_number,user_id,"cv2")
            out_image = cv2.cvtColor(out_image, cv2.COLOR_BGRA2RGBA)
            yield out_image
        for i in range(div_count):
            current_pose_dic["eye"]["left"]=1-i/(div_count/2)
            current_pose_dic["head"]["y"]=3-i*3/(div_count/2)
            current_pose_dic["neck"]=3-i*3/(div_count/2)
            current_pose_dic["body"]["y"]=5-i*5/(div_count/2)
            _,out_image = Thi.inference_dic(current_pose_dic,img_number,user_id,"cv2")
            out_image = cv2.cvtColor(out_image, cv2.COLOR_BGRA2RGBA)
            yield out_image
        for i in range(div_count):
            current_pose_dic["eye"]["left"]=i/div_count
            current_pose_dic["eye"]["right"]=i/div_count
            current_pose_dic["head"]["y"]=-3+i*3/(div_count/2)
            current_pose_dic["neck"]=-3+i*3/(div_count/2)
            current_pose_dic["body"]["z"]=i*3/div_count
            current_pose_dic["body"]["y"]=-5+i*5/(div_count/2)
            _,out_image = Thi.inference_dic(current_pose_dic,img_number,user_id,"cv2")
            out_image = cv2.cvtColor(out_image, cv2.COLOR_BGRA2RGBA)
            yield out_image 
        for i in range(div_count):
            current_pose_dic["eye"]["left"]=0.0
            current_pose_dic["eye"]["right"]=0.0
            current_pose_dic["head"]["y"]=3-i*3/(div_count/2)
            current_pose_dic["neck"]=3-i*3/(div_count/2)
            current_pose_dic["body"]["z"]=3-i*3/div_count
            current_pose_dic["body"]["y"]=5-i*5/(div_count/2)
            _,out_image = Thi.inference_dic(current_pose_dic,img_number,user_id,"cv2")
            out_image = cv2.cvtColor(out_image, cv2.COLOR_BGRA2RGBA)
            yield out_image

#サンプル 9　inference_img() poseはパック形式形式をリポジトリの形式に変換 イメージは事前ロード,パック形式で連続変化させる  画像＝cv2形式
def test9(Thi,img_number,pose_dic):
        user_id=0
        for i in range(50):
            packed_current_pose=[
                "happy",[0.5,0.0],"wink", [i/50,0.0], [0.0,0.0], [0.0,0.0],"ooo", [0.0,0.0], [0.0,i*3/50],i*3/50, 0.0, 0.0, 0.0]
            current_pose=Thi.get_pose(packed_current_pose) #packed_pose=>current_pose2
            _,out_image=Thi.inference_img(current_pose,img_number,user_id,"cv2")
            out_image = cv2.cvtColor(out_image, cv2.COLOR_BGRA2RGBA)
            yield out_image 
            
        for i in range(100):
            packed_current_pose=[
                "happy", [0.5,0.0], "wink",[1-i/50,0.0], [0.0,0.0], [0.0,0.0], "ooo", [0.0,0.0], [0.0,3-i*3/50], 3-i*3/50, 0.0, 0.0, 0.0,]
            current_pose2=Thi.get_pose(packed_current_pose)#packed_current_pose==>current_pose2
            _,out_image=Thi.inference_img(current_pose2,img_number,user_id,"cv2")
            out_image = cv2.cvtColor(out_image, cv2.COLOR_BGRA2RGBA)
            yield out_image 
            
        for i in range(100):
            packed_current_pose=[
                "happy", [0.5,0.0], "wink", [i/100,i/100], [0.0,0.0], [0.0,0.0], "ooo", [0.0,0.0], [0.0,-3+i*3/50], -3+i*3/50,0.0, 0.0,0.0,]
            current_pose2=Thi.get_pose(packed_current_pose)#packed_current_pose==>current_pose2
            _,out_image=Thi.inference_img(current_pose2,img_number,user_id,"cv2")
            out_image = cv2.cvtColor(out_image, cv2.COLOR_BGRA2RGBA)
            yield out_image 
            
        for i in range(100):
            packed_current_pose=[
                "happy", [0.5,0.0], "wink", [0.0,0.0], [0.0,0.0], [0.0,0.0], "ooo",  [0.0,0.0], [0.0,3-i*3/100],  3-i*3/100,  0.0, 0.0, 0.0,]
            current_pose2=Thi.get_pose(packed_current_pose) #packed_current_pose==>current_pose2
            _,out_image=Thi.inference_img(current_pose2,img_number,user_id,"cv2")
            out_image = cv2.cvtColor(out_image, cv2.COLOR_BGRA2RGBA)
            yield out_image 

        print("TEST10 END")            

