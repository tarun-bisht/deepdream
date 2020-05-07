from DeepDreaming import DeepDream,array_to_img,load_image
import os
import argparse

parser=argparse.ArgumentParser(description="Generate Deep Dream Image Frames for generating video ")
parser.add_argument("-d","--dream",required=True,help="Name of Dream to create. Should be folder name inside dream folder which should contain and img_0.jpg")
parser.add_argument("-mdim","--maxdim",help="Maximum Dimension of dream to create",default=512)
parser.add_argument("-n","--maxframes",help="Number of Image frames to create",default=50)

arg=vars(parser.parse_args())

dream_name=arg["dream"]
dream_path=os.path.join("dream",dream_name)

deep_dream=DeepDream()

max_dim=int(arg["maxdim"])
max_frames=int(arg["maxframes"])

if not os.path.isdir("dream"):
    os.mkdir("dream") 
    print("Dream Folder Created Successfully but is empty right now.")
    assert("dream folder is empty. create a folder with dream name and add an image with name img_0.jpg inside it.")
    exit()

if not os.path.isdir(os.path.join("dream",dream_name)) or not os.path.isfile(os.path.join("dream",dream_name,"img_0.jpg")):
    assert("dream folder is empty. create a folder with dream name and add an image with name img_0.jpg inside it.")
    exit()

y_size,x_size=load_image(os.path.join(dream_path,f"img_0.jpg"),max_dim=max_dim).shape[1:-1]
print(f'({x_size},{y_size})')

for i in range(0,max_frames):
    print("Iteration: ",i+1)
    if os.path.isfile(os.path.join(dream_path,f"img_{i+1}.jpg")):
        print("moving forward! Already Exists")
    else:
        image_array=load_image(os.path.join(dream_path,f"img_{i}.jpg"),max_dim=max_dim)
        image_array=deep_dream.run_deep_dream(image_array,num_octaves=1,steps_per_octave=5)
        x_trim=1
        y_trim=1
        image_array=image_array[:,0+x_trim:y_size-y_trim,0+y_trim:x_size-x_trim,:]
        image=array_to_img(image_array,deprocessing=True)
        image=image.resize((x_size,y_size))
        image.save(os.path.join(dream_path,f"img_{i+1}.jpg"),mode="RGB")

