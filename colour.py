# import cv2
# import pandas as pd

# video_path = 'input2.mp4'
# cap = cv2.VideoCapture(video_path)

# # declaring global variables (are used later on)
# clicked = False
# r = g = b = x_pos = y_pos = 0

# # Reading csv file with pandas and giving names to each column
# index = ["color", "color_name", "hex", "R", "G", "B"]
# csv = pd.read_csv('Color-Detection-OpenCV-main\colors.csv', names=index, header=None)


#     # function to calculate minimum distance from all colors and get the most matching color
# def get_color_name(R, G, B):
#     minimum = 10000
#     for i in range(len(csv)):
#         d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
#         if d <= minimum:
#             minimum = d
#             cname = csv.loc[i, "color_name"]
#     return cname


# # function to get x,y coordinates of mouse double click
# def draw_function(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         global b, g, r, x_pos, y_pos, clicked
#         clicked = True
#         x_pos = x
#         y_pos = y
#         b, g, r = frame[y, x]
#         b = int(b)
#         g = int(g)
#         r = int(r)


# cv2.namedWindow('image')
# cv2.setMouseCallback('image', draw_function)


# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
    

#     cv2.imshow("image", frame)
#     if clicked:

#         # cv2.rectangle(image, start point, endpoint, color, thickness)-1 fills entire rectangle
#         cv2.rectangle(frame, (20, 20), (750, 60), (b, g, r), -1)

#         # Creating text string to display( Color name and RGB values )
#         text = get_color_name(r, g, b) + ' R=' + str(r) + ' G=' + str(g) + ' B=' + str(b)

#         # cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType )
#         cv2.putText(frame, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

#         # For very light colours we will display text in black colour
#         if r + g + b >= 600:
#             cv2.putText(frame, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
#             print(text)

#         clicked = False
#     # cv2.imshow("frame",frame)
#     # Break the loop when user hits 'esc' key
#     if cv2.waitKey(20) & 0xFF == ord("q"):
#         break
# cap.release()
# cv2.destroyAllWindows()


# # new

# import cv2
# import pandas as pd
# import os
# import sqlite3

# # Path to the folder containing images
# image_folder_path = 'path/to/images'

# # Reading csv file with pandas and giving names to each column
# index = ["color", "color_name", "hex", "R", "G", "B"]
# csv = pd.read_csv('Color-Detection-OpenCV-main\colors.csv', names=index, header=None)

# # Connect to SQLite database
# conn = sqlite3.connect('color_database.db')
# cursor = conn.cursor()

# # Create a table to store color information if it doesn't exist
# cursor.execute('''
#     CREATE TABLE IF NOT EXISTS colors (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         image_path TEXT,
#         center_color_name TEXT,
#         center_color_rgb TEXT
#     )
# ''')
# conn.commit()

# # Function to calculate minimum distance from all colors and get the most matching color
# def get_color_name(R, G, B):
#     minimum = 10000
#     cname = None
#     for i in range(len(csv)):
#         d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
#         if d <= minimum:
#             minimum = d
#             cname = csv.loc[i, "color_name"]
#     return cname

# def process_image(image_path):
#     # Read the image
#     img = cv2.imread(image_path)

#     # Get image dimensions
#     height, width, _ = img.shape

#     # Get center coordinates
#     center_x, center_y = width // 2, height // 2

#     # Get color at center coordinates
#     b, g, r = img[center_y, center_x]

#     # Calculate color name
#     color_name = get_color_name(r, g, b)

#     # Store color information in the database
#     cursor.execute('''
#         INSERT INTO colors (image_path, center_color_name, center_color_rgb)
#         VALUES (?, ?, ?)
#     ''', (image_path, color_name, f'R={r} G={g} B={b}'))
#     conn.commit()

#     # Display color information at the center coordinates
#     cv2.rectangle(img, (center_x - 50, center_y - 50), (center_x + 50, center_y + 50), (b, g, r), -1)
#     text = f'{color_name} R={r} G={g} B={b}'
#     cv2.putText(img, text, (center_x - 40, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

#     # Display the image
#     cv2.imshow("image", img)
#     cv2.waitKey(0)

# # Process each image in the folder
# for filename in os.listdir(image_folder_path):
#     if filename.endswith(('.jpg', '.png', '.jpeg')):
#         image_path = os.path.join(image_folder_path, filename)
#         process_image(image_path)

# # Close the database connection
# conn.close()
# cv2.destroyAllWindows()


# # def get_color_name(R, G, B):
# #     minimum = 10000
# #     cname = None
# #     for i in range(len(csv)):
# #         d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
# #         if d <= minimum:
# #             minimum = d
# #             cname = csv.loc[i, "colour"]
# #     return cname




# import cv2
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
# import numpy as np

# # Load pre-trained ResNet50 model
# model = ResNet50(weights='imagenet')

# def predict_body_type(image_path):
#     # Load and preprocess the image
#     img = image.load_img(image_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)

#     # Make predictions
#     predictions = model.predict(img_array)

#     # Decode predictions
#     decoded_predictions = decode_predictions(predictions)

#     # Print the top prediction
#     print("Predictions:")
#     for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
#         print(f"{i + 1}: {label} ({score:.2f})")

# # Example usage
# image_path = "path/to/your/car/image.jpg"
# predict_body_type(image_path)



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
