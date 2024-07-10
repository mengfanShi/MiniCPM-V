import requests
import json
import cv2
from PIL import Image
import gradio as gr
import base64
from io import BytesIO

def convert_image_to_base64(images):
    base64_list = []
    for image in images:
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        base64_list.append(image_base64)
    return base64_list

def test_image(image, model_name, url, question=None):
    image_base64 = convert_image_to_base64([image])
    headers = {"Content-Type": 'application/json'}
    body = {
        'image_base64_list': image_base64,
        'question': question,
        'model': model_name,
    }

    response = requests.post(url, data=json.dumps(body), headers=headers)
    obj = response.json()

    answer = obj["answer"]
    txt = ''
    if len(answer) >= 1:
        txt += ('图像描述: ' + answer[0] + '\n')
        if len(answer) > 1:
            txt += ('问题: ' + question + '\n')
            txt += ('答案: ' + answer[1])
    return txt

def test_video(video_path, model_name, url, question=None):
    images = extract_frames(video_path, num_frame=8)
    image_base64_list = convert_image_to_base64(images)

    headers = {"Content-Type": 'application/json'}
    body = {
        'image_base64_list': image_base64_list,
        'question': question,
        'model': model_name,
    }

    response = requests.post(url, data=json.dumps(body), headers=headers)
    obj = response.json()

    answer = obj["answer"]
    txt = ''
    if len(answer) >= 1:
        txt += ('视频描述: ' + answer[0] + '\n')
        if len(answer) > 1:
            txt += ('问题: ' + question + '\n')
            txt += ('答案: ' + answer[1])
    return txt

def extract_frames(video_path, num_frame=8):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = num_frames // num_frame
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            frames.append(pil_image)
        frame_count += 1

    cap.release()
    return frames


model_name = 'minicpm-2.5'
url = "http://0.0.0.0:8888/upload"

with gr.Blocks() as demo:
    gr.Markdown("# 多模态图像描述")
    model_options = ["minicpm-2.5", "minicpm-2.5-int4"]
    model_name = gr.Dropdown(choices=model_options, label="选择模型")

    with gr.Tab("上传图像"):
        image_input = gr.Image(type="pil", label="上传图像")
        question_input = gr.Textbox(label="输入问题 (可选)")
        image_output = gr.Textbox(label="描述输出")
        image_button = gr.Button("获取图像描述")

        def process_image(image, model_name, question):
            question = None if question == '' else question
            return test_image(image, model_name, url, question)

        image_button.click(process_image, inputs=[image_input, model_name, question_input], outputs=image_output)

    with gr.Tab("上传视频"):
        video_input = gr.Video(label="上传视频")
        video_question_input = gr.Textbox(label="输入问题 (可选)")
        video_output = gr.Textbox(label="描述输出")
        video_button = gr.Button("获取视频描述")

        def process_video(video, model_name, question):
            question = None if question == '' else question
            return test_video(video, model_name, url, question)

        video_button.click(process_video, inputs=[video_input, model_name, video_question_input], outputs=video_output)

demo.launch(share=False, server_name="0.0.0.0", server_port=8081)