# this program is fed the raw image files which are to be differentiated between
# positives will formatted into an email and sent to the desired adress with proper authentcation
# DATA for this program was pulled from an image database so users will have to integrate location of there data
# an HTML template was used as formatting but is not necessary

# @chasealbright


import numpy as np
import os
import cv2
import pandas as pd
import csv
from os import listdir
from os.path import isfile, join
from urllib.request import urlopen
from urllib.parse import urljoin
import tensorflow
import random
KERAS_BACKEND=tensorflow
import keras
from datetime import date
import datetime
import time
from datetime import timedelta
import h5py
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import RMSprop, Adam, Adadelta
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from operator import itemgetter
import configparser
from bs4 import BeautifulSoup
import smtplib
import os.path as op
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import encoders

def parseParams(filename):
    parser = configparser.ConfigParser()
    parser.read(filename)
    params = {"imageWidth": parser.get('Image_Settings', 'imageWidth'),
              "imageHeight": parser.get('Image_Settings', 'imageHeight'),
              "imageDownsizeFactor": parser.get('Model_Settings', 'imageDownsizeFactor'),
              "nb_filters": parser.get('Model_Settings', 'nb_filters'),
              "nb_conv": parser.get('Model_Settings', 'nb_conv'),
              "nb_epoch": parser.get('Model_Settings', 'nb_epoch'),
              "batch_size": parser.get('Model_Settings', 'batch_size'),
              "outputFolderName": parser.get('System_Settings', 'outputFolderName'),
              "modelFolderName": parser.get('System_Settings', 'modelFolderName'),
              "basefilename": parser.get('System_Settings', 'basefilename'),
              "truckThreshold": parser.get('Model_Settings', 'truckThreshold')}
    return params
def fetchPrevDayFiles(cam, day):
    url = "http://192.168.38.202/na_viscam/" 
    url += day.strftime('%Y%m%d') + "/" + str(cam) + "/" 
    website = urlopen(url)
    urlList = []
    html = website.read().decode('utf-8')
    soup = BeautifulSoup(html, "lxml")
    soup.prettify()
    for anchor in soup.findAll('a', href=True):
        if '.jpg' in anchor['href']:
            if urljoin(url, anchor['href']) not in urlList:
                urlList.append(urljoin(url, anchor['href']))      
        
    return urlList
def findTruckCandidateImages(urlList, threshold, model):
    truckUrlList = []
    i = 0
    for image_path in urlList:
        i=i+1
        resp = urlopen(image_path)
        img = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.cvtColor(cv2.imdecode(img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (image_size_c, image_size_r))
        img = np.expand_dims(img,axis=3)
        img = np.expand_dims(img,axis=0)

        result = model.predict(img)
        
        if(i<20):
          print(result)
        if result > threshold:
            truckUrlList.append(image_path)
        
    return truckUrlList  
def createHtmlPage(urlList):
    
    page_template = """<!DOCTYPE html>
        <html>
        <head></head>
        <body>
        <h1>Freemont</h1>
        <p>Images</p>
        <div>
           <button onclick="prev()">Prev</button>
           <button onclick="next()">Next</button>
        <div class="gallery">       
            <img id="image-viewer" src="">
        </div>
        <script>
                var imagesUrls = [%IMAGES%]
                var current = 0;
                function setImage(imageUrl){
                  document.getElementById('image-viewer').src = imageUrl;
                }
                
                setImage(imagesUrls[0]);
                
                function prev(){
                 if(current > 0){
                   var url = imagesUrls[--current];
                   setImage(url);
                 }
                }
                
                
                function next(){
                  if(current < imagesUrls.length - 1){
                    var url = imagesUrls[++current];
                    setImage(url);
                  }
               }
         </script>
        </body>
        </html>
"""
    images = ",".join(['"{}"'.format(url) for url in urlList])
    return page_template.replace("%IMAGES%", images)
def sendEmailMessage(originAddress, destinationAddresses, htmlPage, htmlPageName):
    msg = MIMEMultipart()
    msg['From'] = originAddress
    msg['To'] = COMMASPACE.join(destinationAddresses)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = "Today's Truck HTML"

    msg.attach(MIMEText('Find your trucks attached'))


    part = MIMEBase('application', "octet-stream")
    part.set_payload(htmlPage)
    encoders.encode_base64(part)
    part.add_header('Content-Disposition',
                    'attachment; filename="{}"'.format(htmlPageName))
    msg.attach(part)

    smtp = smtplib.SMTP('webmail.mailinglist.com')
    smtp.starttls()
    #smtp.login(username, password)
    smtp.sendmail(originAddress, destinationAddresses, msg.as_string())
    smtp.quit()
    
if __name__ == "__main__":
 
  yesterday = datetime.datetime.now() - timedelta(1)
  model = load_model(r'/dbfs/FreemontModel.h5')
  fullUrlList = fetchPrevDayFiles(1096, yesterday)
  image_size_c = 640
  image_size_r = 360
  truckCandidateImages = findTruckCandidateImages(fullUrlList, 0.5, model)
  print(len(fullUrlList))
  print(len(truckCandidateImages))
  htmlPage = createHtmlPage(truckCandidateImages)
  sendEmailMessage('user@email.com',['user@email.com'], htmlPage, yesterday.strftime('%Y%m%d') + '.html')