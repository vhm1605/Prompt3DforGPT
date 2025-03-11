import cv2
import base64
from openai import OpenAI, AzureOpenAI
import os
import numpy as np
import json
import dotenv
import time
import argparse
import openai
import torch
from dotenv import load_dotenv
from groq import Groq

import os

load_dotenv(dotenv_path= "../.env")

API_KEY = os.getenv("OPEN_API_KEY")
# Thay 'your_groq_api_key_here' bằng API key thực tế của bạn
#os.environ['GROQ_API_KEY'] = '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'
os.environ['OPENAI_API_KEY'] = API_KEY


def reshape_depth_3D(video, new_depth=200):
  temp = torch.tensor(video).float()
  height = temp.shape[2]
  width = temp.shape[3]
  temp = temp.permute(0, 4, 1, 2, 3)
  resized_temp = torch.nn.functional.interpolate(temp, size=(new_depth, height, width), mode='trilinear', align_corners=False)
 # print(resized_temp.shape)
  return resized_temp.permute(0, 2, 3, 4, 1).numpy()


def image_resize_for_vlm(frame, size_frame = (256, 256), inter=cv2.INTER_AREA):
    resized_frame = cv2.resize(
        frame, size_frame, interpolation=inter)
    return resized_frame



# Create a grid of frames
import numpy as np
import cv2

def create_frame_grid(video_matrix, start_indice, grid_size=5, render_pos='topright'):
    spacer = 0
    num_frames = grid_size ** 2

    frames = video_matrix[start_indice:start_indice + num_frames]
    actual_indices = list(range(start_indice, start_indice + num_frames))

    if len(frames) < num_frames:
        raise ValueError("Not enough frames to create the grid.")

    frame_height, frame_width = frames[0].shape[:2]

    grid_height = grid_size * frame_height + (grid_size - 1) * spacer
    grid_width = grid_size * frame_width + (grid_size - 1) * spacer

    grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

    for i in range(grid_size):
        for j in range(grid_size):
            index = i * grid_size + j
            frame = frames[index].copy()
            cX, cY = frame.shape[1] // 2, frame.shape[0] // 2
            max_dim = int(min(frame.shape[:2]) * 0.3)
            overlay = frame.copy()

            if render_pos == 'center':
                circle_center = (cX, cY)
            else:
                circle_center = (frame.shape[1] - max_dim // 2, max_dim // 2)

            # Vẽ hình tròn với độ trong suốt
            cv2.circle(overlay, circle_center, max_dim // 2, (255, 255, 255), -1)
            alpha = 0.3
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.circle(frame, circle_center, max_dim // 2, (255, 255, 255), 2)

            # Tính toán vị trí văn bản để nằm chính giữa hình tròn
            font_scale = max_dim / 70
            text = str(index + start_indice + 1)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            
            text_x = circle_center[0] - text_size[0] // 2
            text_y = circle_center[1] + text_size[1] // 2

            # Vẽ số lên hình ảnh
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 5)

            # Gán vào ảnh lưới
            y1 = i * (frame_height + spacer)
            y2 = y1 + frame_height
            x1 = j * (frame_width + spacer)
            x2 = x1 + frame_width
            grid_img[y1:y2, x1:x2] = frame

    return grid_img, actual_indices



def stack_img_understanding(stack_img, prompt_message,  size_frame = (256, 256)):
  PROMPT_MESSAGES = [
     {
            "role": "system",
            "content": "You are an AI medical assistant specialized in analyzing medical images and providing diagnostic insights."
        },
      {
          "role": "user",
          "content": [
              {
                  "type": "text",
                  "text": prompt_message
              },
          ]
      },
  ]
  total_frames = len(stack_img)
  for i in range(total_frames):
    frame = stack_img[i]
    frame = image_resize_for_vlm(frame,  size_frame = size_frame)
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frame = base64.b64encode(buffer).decode("utf-8")
    my_dict = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64Frame}",
            "detail": "high"
        },
    }
    PROMPT_MESSAGES[1]['content'].append(my_dict)
  #   break
  # client = Groq(
  #       api_key=os.environ.get("GROQ_API_KEY"),
  #   )
  # completion = client.chat.completions.create(
  #     model="llama-3.2-11b-vision-preview",
  #     messages=PROMPT_MESSAGES,
  #     temperature=1,
  #     max_completion_tokens=1024,
  #     top_p=1,
  #     stream=False,
  #     stop=None,
  # )
  



  client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
  completion = client.chat.completions.create(
  model="gpt-4o",
  messages=PROMPT_MESSAGES,
  max_tokens=500,
  temperature=0.2,
)

  return completion.choices[0].message.content





def prompt_3D_image(video_path, grid_size = 5, new_depth = 200,  size_frame = (512, 512)):
  prompt_message = (
      f"bạn được cung cấp ảnh 3D đầu vào (ảnh chụp 1 trong 3 bộ phận: đầu cổ, bụng xương chậu, ngực hoặc nó chụp toàn thân)(dưới định dạng được chia thành nhiều ảnh 2D, mỗi ảnh 2D có 1 số thể hiện thứ tự của ảnh 2D đó trong ảnh 3D). Hãy chẩn đoán, phân tích y khoa các bộ phận trong ảnh tôi gửi và các điểm bất thường trong ảnh theo format sau:\n Đây là ảnh của vùng....\n ---\n*Báo cáo Chẩn đoán Y khoa:*\n...--- "
      
  )
  img_3D_array = np.load(video_path)
  X_min = img_3D_array.min()
  X_max = img_3D_array.max()
 # print(X_min, X_max)
  img_3D_scaled = (img_3D_array - X_min) / (X_max - X_min) * 255
  img_3D_scaled = img_3D_scaled.astype(np.uint8)
  img_3D_array = img_3D_scaled
  X_min = img_3D_array.min()
  X_max = img_3D_array.max()
 # print(X_min, X_max)
  video_matrix = img_3D_array.reshape(1, *img_3D_array.shape, 1)
  video_matrix = reshape_depth_3D(video_matrix, new_depth = new_depth)
  video_matrix = video_matrix.squeeze()
 # print(video_matrix.shape)
  if len(video_matrix.shape)==3:
    video_matrix = np.stack([video_matrix] * 3, axis=-1)
  #print(video_matrix.shape)
  num_slice = grid_size**2
  stack_img = []
  num_grid = video_matrix.shape[0]//num_slice
  
  for i in range(num_grid):
    start_indice = i * num_slice
    image, used_frame_indices = create_frame_grid(video_matrix, start_indice, grid_size=grid_size)
    stack_img.append(image)
    cv2.imwrite(
                os.path.join(
                    '.\Test',
                    f"grid_image_sample{i}.png"),
                image)
  description = stack_img_understanding(stack_img, prompt_message, size_frame = size_frame)
  print(description)


folder_path = '.\Test'

# Lấy danh sách các tệp trong thư mục
files = os.listdir(folder_path)

# Duyệt qua từng tệp và xóa
for file in files:
    file_path = os.path.join(folder_path, file)
    if os.path.isfile(file_path):
        os.remove(file_path)
prompt_3D_image('_abdomen_pelvis_day_3_patient_64.npy')








