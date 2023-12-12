from PIL import Image
import numpy as np
from collections import Counter
import os 
import pandas as pd

# very specific colour dataset 
index =  ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('Color-Detection-OpenCV-main\colors.csv', names=index, header=None) 

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


for filename in os.listdir(image_folder):

  image_path = os.path.join(image_folder, filename)

  dominant_color = color(Image.open(image_path))
  print(f"Image: {filename} - color: {dominant_color}")
