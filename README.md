# deepdream
Create deepdream images in python.
# Results
[![deepdream result videp](http://img.youtube.com/vi/wmDjFQDh5BY/0.jpg)](http://www.youtube.com/watch?v=wmDjFQDh5BY)
[![deepdream result videp](http://img.youtube.com/vi/wmDjFQDh5BY/0.jpg)](http://www.youtube.com/watch?v=wmDjFQDh5BY)
[![deepdream result videp](http://img.youtube.com/vi/wmDjFQDh5BY/0.jpg)](http://www.youtube.com/watch?v=wmDjFQDh5BY)

# Installation
- Make sure python is installed on your system if not then download [python](https://www.python.org/downloads/)
- clone or download and extract this repository
- Install all dependencies shown below using terminal or command prompt via pip

### Dependencies

#### Windows
1. open-cv `pip install opencv-contrib-python`
2. numpy `pip install numpy`
3. tensorflow `pip install tensorflow-gpu` or `pip install tensorflow`
4. pillow `pip install pillow`

#### Linux
1. open-cv `pip3 install opencv-contrib-python`
2. numpy `pip3 install numpy`
3. tensorflow `pip3 install tensorflow-gpu` or `pip3 install tensorflow`
4. pillow `pip3 install pillow`

# Creating DeepDream image frames
- Inside `deepdream/dreams` create a new folder in that path and name it with your dream name you are creating.
- put a image inside that folder whose dream is to be created and name it `img_0.jpg`
- from root folder of this repository run python script `deep_framed.py`
    ```bash
        python deep_frames.py -d=" your dream name "
    ```
    arguments:
    1. `--dream` or `-d` : Name of Dream to create. Should be folder name inside dream folder which should contain and img_0.jpg
    2. `-mdim` or `--maxdim` : Maximum Dimension of dream to create. default is 512
    3. `-n` or `--maxframes` : Number of Image frames to create. default is 50 ie.. by default it will create 50 deepdream image frames inside your dream folder.
    example usage:
    ```bash
        python deep_frames.py -d=" your dream name " -n=100 -mdim=1024
    ```
# Creating Videos from deep image frames
from root folder of this repository run python script `frames_to_video.py`
```bash
    python frames_to_video.py -d=" your dream name "
```
arguments:
1. `--dream` or `-d` : Name of Dream to create. Should be folder name inside dream folder.
2. `-f` or `--framerate` : Frame Rate of Video. default is 20. How fast you want to your video to progress?
example usage:
```bash
    python frames_to_video.py -d=" your dream name " -f=10
```

# License
[MIT](https://github.com/tarun-bisht/deepdream/blob/master/LICENSE)
 
