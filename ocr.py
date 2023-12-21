#MAIN CODE CONTAINING OCR AND COLOUR DETECTION

import os 
from paddleocr import PaddleOCR
import cv2
import psycopg2
import datetime
import time
import pandas as pd
import numpy as np
from collections import Counter


# connecting to the database
connection = psycopg2.connect(user="postgres",
                              password="",
                              host="127.0.0.1",
                              port="5432",
                              database="traffic_violation")

# input video
input = 'car_plates3'
numbers = []
# ocr for detection
ocr = PaddleOCR(lang='en')


# very specific colour dataset 
index =  ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('Color-Detection-OpenCV-main\colors.csv', names=index, header=None)\

# # alternate  dataset
# index = [ "R", "G", "B","colour"]
# csv = pd.read_csv('colour.csv', names=index, header=None)

def color(image):
  image_array = np.array(image)
  pixels = image_array.reshape(-1, 3)
  
#   counting the number of times each colour combo is seen in the image
  colour_counts = Counter(tuple(color) for color in pixels)

#   finds the colour which appears the most
  main_colour = colour_counts.most_common(1)[0][0]

# compares the most appeared colour with the dataset to get the name of colour
  colour_differences = [
        abs(main_colour[0] - int(row["R"])) + abs(main_colour[1] - int(row["G"])) + abs(main_colour[2] - int(row["B"]))
        for _, row in csv.iterrows()
    ]
  min_difference = colour_differences.index(min(colour_differences))
  
  return csv.loc[min_difference, "color_name"]


image_folder = "car_plates3"


# applying ocr on captured images and saving to database 
for filename in os.listdir(input):
    if filename.endswith((".jpg")):
        image_path = os.path.join(f'{input}\\{filename}')
        image = cv2.imread(image_path)
        result = ocr.ocr(image)
        dominant_colour = color(image)
        type_of_car = "nothing"
        
        try:
            number_plate = result[0][0][1][0]
            
            with connection.cursor() as inner_cur:
                
                inner_cur.execute("INSERT INTO number_plates (license_plate, todays_date, time_of_violation, colour, type_of_car, image_name) values (%s,%s,%s,%s,%s,%s);" , (number_plate,datetime.date.today(),time.strftime("%H:%M:%S", time.localtime()),dominant_colour,type_of_car,image_path))
                inner_cur.execute("SELECT * FROM number_plates")
                rows = inner_cur.fetchall()
                for row in rows :
                    a = row[0]
                    numbers.append(a) 
                    
                # Deleting duplicates of the number plates
                if numbers.count(number_plate) > 1:
                    inner_cur.execute("DELETE FROM number_plates WHERE license_plate = %s AND ctid NOT IN (SELECT min(ctid) FROM number_plates WHERE license_plate = %s);", (number_plate, number_plate))
            connection.commit()
        except TypeError :
            continue


# NOTE THAT SOME NUMBER PLATE SEEM LIKE THEY HAVE BEEN DUPLICATED BECAUSE THE INPUT VIDEO HAS NUMBER PLATES IN A DIFFERENT LANGUAGE ( I THINK CHINESE AND ENGLISH
# SCREENSHOT OF THE CONTENTS OF THE DATABASE IS ALSO UPLOADED 
            
