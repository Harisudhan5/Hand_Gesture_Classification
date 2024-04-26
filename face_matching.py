from deepface import DeepFace
import os 
par = "Target_Faces"
img1 = "Target_Faces\Harisudhan.jpg"
for file in os.listdir('Target_Faces'):
    img2 = os.path.join(par,file)
    resp = DeepFace.verify(img1_path = img1, img2_path = img2, model_name = "Facenet")
    print("Response for image matching =", resp)
    print(type(img2))
    p = str(file).split('.')
    print("FileName = ",p[0])

