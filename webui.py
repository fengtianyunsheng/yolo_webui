import cv2
from ultralytics import YOLO
from datetime import datetime
import json
import gradio as gr
import os

os.environ['GRADIO_TEMP_DIR'] = 'temp'


# 定义保存用户输入的函数
def save_inputs(model,output_image_path):
    data = {
        'model':model,
        'output_image_path':output_image_path,
    }
    with open('webui_user_data.json', 'w') as f:
        json.dump(data, f)

# 定义加载用户输入的函数
def load_user_data():
    try:
        with open('webui_user_data.json', 'r') as f:
            data = json.load(f)
            model = data.get('model','')
            # source = data.get('source', '')
            output_image_path = data.get('output_image_path', 'output')
        return model,output_image_path
            
        
    except FileNotFoundError:
        return None

# 加载 YOLOv9 模型
def run(model,source,output_image_path):
  save_inputs(model,output_image_path)
  model = YOLO(model)  # 替换为你实际的 YOLOv9 模型路径

  # 读取图像
  image_path = source
  image = cv2.imread(image_path)

  # 进行目标检测
  results = model(image)

  # 解析检测结果
  detected_objects = {}
  for result in results:
      for box in result.boxes:
          class_id = int(box.cls)
          class_name = model.names[class_id]
          if class_name in detected_objects:
              detected_objects[class_name] += 1
          else:
              detected_objects[class_name] = 1

  # 输出检测结果
  output_string = ' '.join([f"{count} {label}" for label, count in detected_objects.items()])
  print(output_string)

  # 可视化检测结果（可选）
  for result in results:
      for box in result.boxes:
          x1, y1, x2, y2 = map(int, box.xyxy[0])
          class_id = int(box.cls)
          class_name = model.names[class_id]
          cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
          cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

  # 保存结果图像
  # 获取当前时间
  now = datetime.now()
  # 格式化时间字符串
  formatted_time = now.strftime('%Y%m%d%H%M%S')
  if not os.path.exists(output_image_path):
        os.makedirs(output_image_path)
  save_dir = output_image_path+'\\'+str(formatted_time)+'.jpg'

  cv2.imwrite(save_dir, image)
  print(f"Result image saved to {output_image_path}")

  return save_dir,output_string
try:
  model,output_image_path = load_user_data()
except:
  model = "yolo11x.pt"  
  output_image_path = "output"



demo = gr.Interface(
    fn=run,
    inputs=[gr.File(label="选择模型文件",value=model),gr.Image(type="filepath") ,gr.Textbox(label="输入保存目录",value=output_image_path)],
    # inputs=[gr.File(label="选择模型文件"),gr.Image(type="filepath") ,gr.Textbox(label="输入保存目录")],
    outputs=[gr.Image(label="结果图片"),gr.Textbox(label="识别结果")],

)



demo.launch(inbrowser=True)

# 显示结果图像（可选）
# cv2.imshow('Result', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()