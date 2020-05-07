import cv2
import os
import argparse

parser=argparse.ArgumentParser(description="Convert Deep Dream Frame to Video")
parser.add_argument("-d","--dream",required=True,help="Name of Dream to create. Should be folder name inside dream folder")
parser.add_argument("-f","--framerate",help="Frame Rate of Video",default=20)

arg=vars(parser.parse_args())

dream_name=arg["dream"]
dream_path=os.path.join("dream",dream_name)

img_frames = len([f for f in os.listdir(dream_path) if os.path.isfile(os.path.join(dream_path, f))])
print("Number of Video Frames: ",img_frames)
x=0
y=0

if os.path.isfile(os.path.join(dream_path,f"img_0.jpg")):
    y,x=cv2.imread(os.path.join(dream_path,f"img_0.jpg")).shape[:-1]

if os.path.isfile(os.path.join(dream_path,f"img_0.jpg")) and os.path.isfile(os.path.join(dream_path,f"img_1.jpg")):
    y0,x0=cv2.imread(os.path.join(dream_path,f"img_0.jpg")).shape[:-1]
    y1,x1=cv2.imread(os.path.join(dream_path,f"img_1.jpg")).shape[:-1]
    if not x0==x1:
        x=x1
    if not y0==y1:
        y=y1
    img_0=cv2.imread(os.path.join(dream_path,f"img_0.jpg"))
    cv2.resize(img_0,(x,y))
    cv2.imwrite(os.path.join(dream_path,f"img_0.jpg"),img_0)

print("Image Shape: ",(x,y))

fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter(f'{dream_name}.avi',fourcc,int(arg["framerate"]),(x,y))

for i in range(img_frames+1):
    print(f"processing img_{i}.jpg ...")
    path=os.path.join(dream_path,f"img_{i}.jpg")
    frame=cv2.imread(path)
    out.write(frame)

out.release()