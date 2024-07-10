import base64, json, gc, os
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from chat import MiniCPMV2_5

app = Flask(__name__)
CORS(app)

# 全局变量来存储当前加载的模型和模型名称
current_model = None
current_model_name = None

def load_model(model_name):
    global current_model, current_model_name
    if current_model_name != model_name:
        if current_model is not None:
            del current_model
            gc.collect()

        model_path = {
            'minicpm-2.5': 'model/MiniCPM-Llama3-V-2_5',
            'minicpm-2.5-int4': 'model/MiniCPM-Llama3-V-2_5-int4'
        }[model_name]
        current_model = MiniCPMV2_5(model_path)
        current_model_name = model_name

def convert_image_to_base64(images):
    base64_list = []
    for image in images:
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        base64_list.append(image_base64)
    return base64_list

def convert_base64_to_image(base64_list):
    images = []
    for base64_string in base64_list:
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_bytes))
        images.append(image)
    return images

def chat_image(model, image_base64, question=None):
    msgs = [{"role": "user", "content": "请详细描述一下图片中的内容"}]
    input = {"image": image_base64, "question": json.dumps(msgs, ensure_ascii=True)}
    answer_describe = model.chat(input)

    if question is not None:
        msgs.append({"role": "assistant", "content": answer_describe})
        msgs.append({"role": "user", "content": question})
        input = {"image": image_base64, "question": json.dumps(msgs, ensure_ascii=True)}
        answer_question = model.chat(input)
        return [answer_describe, answer_question]
    return [answer_describe]

def chat_video(model, image_list, question=None):
    system_prompt = 'Answer in detail'
    prompt = '这些图片是从同一个视频中抽取出来的，请依据这些图片详细描述一下视频的内容.'
    # save_images(image_list, 'tmp/')

    content = []
    content.append(system_prompt)
    for image in image_list:
        content.append(image.resize((256, 256)))
    content.append(prompt)
    msgs = [{'role': 'user', 'content': content}]
    answer_describe = model.chat_msgs(msgs)

    if question is not None:
        msgs.append({'role': 'assistant', 'content': answer_describe})
        msgs.append({'role': 'user', 'content': question})
        answer_question = model.chat_msgs(msgs)
        return [answer_describe, answer_question]
    return [answer_describe]

def save_images(image_list, path):
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(len(image_list)):
        image = image_list[i]
        image.save(path + str(i) + '.png')

@app.route('/upload', methods=['POST'])
def upload_result():
    data = request.json
    image_base64_list = data.get("image_base64_list")
    question = data.get("question", None)
    model_name = data.get("model", 'minicpm-2.5')

    load_model(model_name)
    answer = []

    try:
        if len(image_base64_list) == 1:
            answer = chat_image(current_model, image_base64_list[0], question=question)
        elif len(image_base64_list) > 1:
            answer = chat_video(current_model, convert_base64_to_image(image_base64_list), question=question)
    except Exception as e:
        print(e)

    response_data = {
        "answer": answer,
    }

    return jsonify(response_data), 200


if __name__ == '__main__':
    app.run(debug=False, threaded=True, host='0.0.0.0', port=8888)