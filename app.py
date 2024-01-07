import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input
import requests
from flask import Flask,render_template,request,redirect,url_for
from cloudant.client import Cloudant


app=Flask(__name__)

model=load_model(r"Updated-xception-diabetic-retinopathy.h5")



@app.route('/')
def index():
    return render_template("index.html")

@app.route('/index.html')
def home():
    return render_template("index.html")

@app.route('/register')
def register():
    return render_template("register.html")


@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=image.load_img(filepath,target_size=(64,64))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred=np.argmax(model.predict(x),axis=1)
        index=['Bear','Crow','Elephant','Rat']
        text="The Classified Animal is : " +str(index[pred[0]])
    return text

if __name__=='__main__':
    app.run()