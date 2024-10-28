import gradio as gr
from PIL import Image
from poser_image_2_template_class import Image2form
from poser_client_v1_3_class import TalkingHeadAnimefaceInterface
from tkh_basic_test_def import test1,test2,test3,test4,test5,test6,test7,test8,test8,test9

tkh_url='http://127.0.0.1:8001'
url="http://127.0.0.1:8008/resr_upscal/"
fd_url="http://127.0.0.1:50001"

# クラスのインスタンスを作成
Thi=TalkingHeadAnimefaceInterface(tkh_url)
I2f = Image2form(url)

user_id=0
pose_dic = Thi.get_init_dic()

def upload_and_process_image(file):
    global input_image
    global img_number
    pil_input_image = Image.open(file)

    # 初期処理を行い、結果をグローバル変数に保存
    _, input_image = I2f.image_data_form(pil_input_image, "pil")
    img_number=Thi.load_img(input_image,0)
    return input_image

def submit_process(test_option):
    global input_image
    global img_number

    processed_image = None  # デフォルト値を設定
    # テストオプションに基づいて処理を実行
    if test_option == "TEST1":
        # テスト1の処理（例）
        img=test1(Thi,input_image)
        yield img
    elif test_option == "TEST2":
        img=test2(Thi,input_image)
        yield img
    elif test_option == "TEST3":
        img=test3(Thi,input_image)
        yield img
    elif test_option == "TEST4":
        img=test4(Thi,img_number)
        yield img
    elif test_option == "TEST5":
        for img in test5(Thi,img_number):  # 逐次画像を返す
            yield img
    elif test_option == "TEST6":
        for img in test6(Thi,img_number):  # 逐次画像を返す
            yield img
    elif test_option == "TEST7":
        for img in test7(Thi,img_number):  # 逐次画像を返す
            yield img
    elif test_option == "TEST8":
        for img in test8(Thi,img_number,pose_dic):  # 逐次画像を返す
            yield img
    elif test_option == "TEST9":
        for img in test9(Thi, img_number,pose_dic):  # 逐次画像を返す
            yield img
    else:
        return img
    # 結果の画像を返す
    return processed_image

# GradioのUI設定
with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Input Image")  # 画像アップロード時に処理
            test_option = gr.Radio(choices=["TEST1", "TEST2", "TEST3", "TEST4", "TEST5", "TEST6", "TEST7", "TEST8", "TEST9",], label="Test Option")
            submit_button = gr.Button("Submit")

        with gr.Column():
            output_image = gr.Image(label="Output Image")
    # イベントハンドラ
    image_input.upload(upload_and_process_image, inputs=image_input, outputs=output_image)  # 画像アップロード時に処理
    submit_button.click(submit_process, inputs=[test_option], outputs=output_image)

# GradioのUIを起動
if __name__ == "__main__":
    iface.launch()
