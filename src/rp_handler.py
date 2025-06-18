import runpod
from runpod.serverless.utils import rp_upload
import json
import urllib.request
import urllib.parse
import time
import os
import requests
import base64
from io import BytesIO

# 定义常量，用于配置API检查和轮询的间隔时间、最大尝试次数，以及ComfyUI服务器的地址和是否在每个任务完成后刷新工作进程
COMFY_API_AVAILABLE_INTERVAL_MS = 50  # API检查间隔时间（毫秒）
COMFY_API_AVAILABLE_MAX_RETRIES = 500  # API检查最大尝试次数
COMFY_POLLING_INTERVAL_MS = int(250)  # 轮询间隔时间（毫秒）
COMFY_POLLING_MAX_RETRIES = int(1200)  # 轮询最大尝试次数
COMFY_HOST = "127.0.0.1:8188"  # ComfyUI服务器地址
REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"  # 是否在每个任务完成后刷新工作进程

def validate_input(job_input):
    """
    验证输入数据是否符合要求。
    """
    if job_input is None:
        return None, "Please provide input"  # 如果输入为空，返回错误信息

    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)  # 尝试将输入的字符串解析为JSON
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"  # 如果解析失败，返回错误信息

    workflow = job_input.get("workflow")
    if workflow is None:
        return None, "Missing 'workflow' parameter"  # 如果缺少workflow参数，返回错误信息

    image = job_input.get("image")
    if image is None:
        return {"workflow": workflow, "image": None}, None
    filename = [
            node.get("inputs")
            for node in workflow.values()
            if node.get("class_type") == "LoadImage"
        ][0]["image"]
    output_name = [
            node.get("inputs")
            for node in workflow.values()
            if node.get("class_type") == "SaveImage"
        ][0]["filename_prefix"]

    return {"workflow": workflow, "image": image, "filename": filename,"output_name": output_name}, None  # 返回验证后的数据和无错误信息

def check_server(url, retries=500, delay=50):
    """
    检查服务器是否可达。
    """
    for i in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"runpod-worker-comfy - API is reachable")  # 如果服务器返回200状态码，表示服务器可达
                return True
        except requests.RequestException as e:
            pass  # 如果发生异常，服务器可能未准备好

        time.sleep(delay / 1000)  # 等待指定时间后重试

    print(
        f"runpod-worker-comfy - Failed to connect to server at {url} after {retries} attempts."
    )
    return False  # 如果超过最大尝试次数仍未连接到服务器，返回False

def upload_images(image,filename):
    """
    将Base64编码的图像上传到ComfyUI服务器。
    """

    responses = []
    upload_errors = []

    print(f"runpod-worker-comfy - image(s) upload")

    blob = base64.b64decode(image)  # 将Base64编码的图像数据解码

    files = {
        "image": (filename, BytesIO(blob), "image/png"),  # 准备要上传的图像文件
        "overwrite": (None, "true"),  # 设置覆盖选项
    }

    response = requests.post(f"http://{COMFY_HOST}/upload/image", files=files)  # 发送POST请求上传图像
    if response.status_code != 200:
        upload_errors.append(f"Error uploading {filename}: {response.text}")  # 如果上传失败，记录错误信息
    else:
        responses.append(f"Successfully uploaded {filename}")  # 如果上传成功，记录成功信息

    if upload_errors:
        print(f"runpod-worker-comfy - image(s) upload with errors")
        return {
            "status": "error",
            "message": "Some images failed to upload",
            "details": upload_errors,
        }  # 如果有上传错误，返回错误消息

    print(f"runpod-worker-comfy - image(s) upload complete")
    return {
        "status": "success",
        "message": "All images uploaded successfully",
        "details": responses,
    }  # 如果所有图像上传成功，返回成功消息

def queue_workflow(workflow):
    """
    将工作流发送到ComfyUI服务器进行处理。
    """
    data = json.dumps({"prompt": workflow}).encode("utf-8")  # 将工作流封装在prompt键下，并编码为JSON格式

    req = urllib.request.Request(f"http://{COMFY_HOST}/prompt", data=data)
    return json.loads(urllib.request.urlopen(req).read())  # 发送请求并将响应解析为JSON

def get_history(prompt_id):
    """
    获取指定提示的历史记录。
    """
    with urllib.request.urlopen(f"http://{COMFY_HOST}/history/{prompt_id}") as response:
        return json.loads(response.read())  # 发送请求并解析响应为JSON

def base64_encode(img_path):
    """
    将图像文件进行Base64编码。
    """
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")  # 读取图像文件并进行Base64编码
        return f"{encoded_string}"

def process_output_images(outputs, job_id):
    """
    处理图像生成的输出，根据配置返回结果。
    """
    COMFY_OUTPUT_PATH = os.environ.get("COMFY_OUTPUT_PATH", "/comfyui/output")  # 获取图像输出路径

    output_images = {}

    for node_id, node_output in outputs.items():
        if "images" in node_output:
            for image in node_output["images"]:
                output_images = os.path.join(image["subfolder"], image["filename"])  # 获取生成的图像文件名

    print(f"runpod-worker-comfy - image generation is done")

    local_image_path = f"{COMFY_OUTPUT_PATH}/{output_images}"  # 构造本地图像路径

    print(f"runpod-worker-comfy - {local_image_path}")

    if os.path.exists(local_image_path):
        if os.environ.get("BUCKET_ENDPOINT_URL", False):
            image = rp_upload.upload_image(job_id, local_image_path)  # 如果配置了AWS S3桶，上传图像并返回URL
            print("runpod-worker-comfy - the image was generated and uploaded to AWS S3")
        else:
            image = base64_encode(local_image_path)  # 如果未配置AWS S3桶，将图像编码为Base64字符串
            print("runpod-worker-comfy - the image was generated and converted to base64")

        return {
            "status": "success",
            "message": image,
        }
    else:
        print("runpod-worker-comfy - the image does not exist in the output folder")
        return {
            "status": "error",
            "message": f"the image does not exist in the specified output folder: {local_image_path}%",
        }  # 如果图像文件不存在，返回错误消息

def handler(job):
    """
    主处理函数，负责处理图像生成任务。
    """
    job_input = job["input"]

    validated_data, error_message = validate_input(job_input)
    if error_message:
        return {"error": error_message}  # 如果输入验证失败，返回错误消息

    workflow = validated_data["workflow"]
    if validated_data.get("image"):
        image = validated_data.get("image")
        filename = validated_data.get("filename")
        upload_result = upload_images(image,filename)  # 上传图像

        if upload_result["status"] == "error":
            return upload_result  # 如果上传图像失败，返回错误消息

    try:
        queued_workflow = queue_workflow(workflow)
        prompt_id = queued_workflow["prompt_id"]
        print(f"runpod-worker-comfy - queued workflow with ID {prompt_id}")  # 将工作流排队处理
    except Exception as e:
        return {"error": f"Error queuing workflow: {str(e)}"}  # 如果排队工作流失败，返回错误消息

    print(f"runpod-worker-comfy - wait until image generation is complete")
    retries = 0
    try:
        while retries < 1800:
            history = get_history(prompt_id)

            # Exit the loop if we have found the history
            if prompt_id in history and history[prompt_id].get("outputs"):
                break
            else:
                # Wait before trying again
                time.sleep(COMFY_POLLING_INTERVAL_MS / 1000)
                retries += 1
        else:
            return {"error": "Max retries reached while waiting for image generation"}
    except Exception as e:
        return {"error": f"Error waiting for image generation: {str(e)}"}

    # Get the generated image and return it as URL in an AWS bucket or as base64
    images_result = process_output_images(history[prompt_id].get("outputs"), job["id"])

    result = {**images_result, "status": "success"}

    return result

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})  # 启动无服务器功能