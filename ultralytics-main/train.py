from ultralytics import YOLO

if __name__=='__main__':
   #model = YOLO("yolov8n.pt")
   model = YOLO("yolov9c.yaml")
   #model = YOLO("yolov8-SwinT.yaml")

   results = model.train(data = "myData.yaml", epochs = 300, batch = 16,device = 0 ) #coco128.yaml MyData.yaml