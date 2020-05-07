import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.models import load_model,Model
from PIL import Image
import time

def load_image(image_path,max_dim=512):
    img=Image.open(image_path)
    img=img.convert("RGB")
    img.thumbnail([max_dim,max_dim])
    img=np.array(img,dtype=np.uint8)
    img=np.expand_dims(img,axis=0)
    return inception_v3.preprocess_input(img)

def deprocess_inception_image(img):
    img = 255*(img+1.0)/2.0
    return np.array(img, np.uint8)

def array_to_img(array,deprocessing=False):
    if deprocessing:
        array=deprocess_inception_image(array)
    if np.ndim(array)>3:
        assert array.shape[0]==1
        array=array[0]
    return Image.fromarray(array)

def show_image(img):
    image=array_to_img(img)
    image.show()

class DeepDream:

    def __init__(self,layers_contributions=['mixed3', 'mixed5']):
        inception=inception_v3.InceptionV3(weights="imagenet",include_top=False)
        print("model loaded")
        self.dream_model=self.deep_dream_model(inception,layers_contributions)
        self.model_output= lambda model,inputs:model(inputs)

    def deep_dream_model(self,model,layer_names):
        model.trainable=False
        outputs=[model.get_layer(name).output for name in layer_names]
        new_model=Model(inputs=model.input,outputs=outputs)
        return new_model

    def get_loss(self,activations):
        loss=[]
        for activation in activations:
            loss.append(tf.math.reduce_mean(activation))
        return tf.reduce_sum(loss)

    # Randomly shift the image to avoid tiled boundaries.
    def random_image_tiling(self,img, maxdim):
        shift = tf.random.uniform(shape=[2], minval=-maxdim, maxval=maxdim, dtype=tf.int32)
        shift_r,shift_d=shift
        img_rolled = tf.roll(img, shift=[shift_r,shift_d], axis=[1,0])
        return shift_r, shift_d, img_rolled

    # Deep Dreaming with gradient ascent on loss using image tiles
    def get_loss_and_grads_with_tiling(self,inputs,tile_size=512,total_variation_weight=0.004):
        shift_r,shift_d,tiled_image=self.random_image_tiling(inputs[0],tile_size)
        grads=tf.zeros_like(tiled_image)
        x_range = tf.range(0, tiled_image.shape[0], tile_size)[:-1]
        if not tf.cast(len(x_range), bool):
            x_range= tf.constant([0])
        y_range = tf.range(0, tiled_image.shape[1], tile_size)[:-1]
        if not tf.cast(len(y_range), bool):
            y_range=tf.constant([0])
        for x in x_range:
            for y in y_range:
                with tf.GradientTape() as tape:
                    tape.watch(tiled_image)
                    image_tile= tf.expand_dims(tiled_image[x:x+tile_size, y:y+tile_size],axis=0)
                    activations=self.model_output(self.dream_model,image_tile)
                    loss=self.get_loss(activations)
                    loss=loss+total_variation_weight*tf.image.total_variation(image_tile) 
                grads=grads+tape.gradient(loss,tiled_image)
        grads = tf.roll(grads, shift=[-shift_r,-shift_d], axis=[1,0])
        grads /= tf.math.reduce_std(grads) + 1e-8
        return loss,grads

    def run_deep_dream(self,input_image,num_octaves=2,octave_size=1.3,steps_per_octave=100,tile_size=512,strength=0.01,total_variation_weight=0):
        img=tf.convert_to_tensor(input_image)
        strength=tf.convert_to_tensor(strength)
        assert len(input_image.shape)<=4 or len(input_image.shape)>=3
        if len(input_image.shape)==3:
            base_shape=img.shape[:-1]
        base_shape=img.shape[1:-1]
        start=time.time()
        for n in range(-num_octaves,num_octaves+1):
            print(f'Processing Octave: {n+num_octaves+1}')
            new_shape=tuple([int(dim*(octave_size**n)) for dim in base_shape])
            img=tf.image.resize(img,new_shape)
            step_start_time=time.time()
            for step in range(steps_per_octave):
                print('.',end='')
                loss,grads=self.get_loss_and_grads_with_tiling(img,tile_size,total_variation_weight)
                img = img + grads*strength
                img = tf.clip_by_value(img, -1.0, 1.0)
            step_end_time=time.time()
            print(f"Time: {step_end_time-step_start_time:.1f} sec")
            print("\n")
        end=time.time()
        print(f"Time elapsed: {end-start:.1f} sec")
        return tf.image.resize(img, base_shape).numpy()