from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
# from sklearn.cluster import KMeans
# import os
from sklearn.cluster import MiniBatchKMeans
import requests
import asyncio
import aiohttp
from flask_cors import CORS
from extarcting_the_charecters import main
# from extarcting_the_charecters import main


# from gevent.pywsgi import WSGIServer

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


# image_000 = cv2.imread('./original/000.png')
# image_425 = cv2.imread('./original/425.png')
# image_bb = cv2.imread('./original/bb.png')
# image_bw = cv2.imread('./original/bw.png')
# image_ww = cv2.imread('./original/ww.png')
# image_wb = cv2.imread('./original/wb.png')
# image_c = cv2.imread('./original/c.png')
# image_e = cv2.imread('./original/e.png')
# image_f = cv2.imread('./original/f.png')
# image_t = cv2.imread('./original/t.png')
# image_x = cv2.imread('./original/x.png')
# image_z = cv2.imread('./original/z.png')

# functions starts

def create_disk_kernel(radius):

    # Ensure the radius is a non-negative integer
    radius = max(0, int(radius))

    # Create a grid of coordinates
    x, y = np.ogrid[-radius:radius+1, -radius:radius+1]

    # Create a binary matrix representing a disk
    disk_matrix = x**2 + y**2 <= radius**2

    return disk_matrix.astype(np.uint8)


def imagePreProcess(theOriginalImage,image):

    # image_o = cv2.imread(theOriginalPath)
    image_o = theOriginalImage
    image_path ='./light/1+8.jpg'
    # image_e = cv2.imread(image_path)
    image_e = image

    result = cv2.subtract(image_o, image_e)
    result2 = cv2.subtract(image_e, image_o)

    add = cv2.add(result, result2)
    gray_image = cv2.cvtColor(add, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)

    kernel = create_disk_kernel(1)
    # kernel = disk(2)
    # img = cv2.dilate(binary, kernel, iterations=1)
    img = cv2.erode(binary, kernel, iterations=2)
    img = cv2.dilate(img, kernel, iterations=2)
    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.dilate(img, kernel, iterations=4)
    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.dilate(img, kernel, iterations=2)
    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.dilate(img, kernel, iterations=2)
    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.dilate(img, kernel, iterations=5)
    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.erode(img, kernel, iterations=4)
    kernel = create_disk_kernel(3)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    

    # output_path = f'./newoout/{os.path.basename(image_path)}'
    # cv2.imwrite('aaaaaaaaaaaaaaaaaaa.jpg', img)

    # Find the white color in the original image
    mask = cv2.inRange(img, 200, 255)  # Adjust the lower and upper bounds as needed for your specific case

    # Invert the mask to get the non-white pixels
    mask_inv = cv2.bitwise_not(mask)

    # Convert grayscale img to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Use the mask to get the non-white pixels of the original image
    res1 = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_inv)

    # Ensure that image_e has the same dimensions as img
    image_e = cv2.resize(image_e, (img.shape[1], img.shape[0]))

    # Use the inverted mask to get the white pixels of the edited image
    res2 = cv2.bitwise_and(image_e, image_e, mask=mask)

    # Combine the two results to get the final image
    final = cv2.addWeighted(res1, 1, res2, 1, 0)
       

    image_path ='AAAAAAAAAAAAAfinal_output_path.jpg'
    return final
def get_top_colors(image, top_colors=3, resize_factor=0.3):
    try:
        # Load and resize the image
        image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)

        # Convert the image from BGR to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Reshape the image to be a list of pixels
        pixels = image.reshape(-1, 3)

        # Perform Mini-Batch K-means clustering
        kmeans = MiniBatchKMeans(n_clusters=top_colors, batch_size=1000, random_state=42)
        kmeans.fit(pixels)

        # Get the RGB values of the cluster centers
        colors_in_img = kmeans.cluster_centers_

        # Convert the colors to integers
        colors_in_img = colors_in_img.round(0).astype(int)
        # print(colors_in_img)

        # Remove colors similar to black or less than [10, 10, 10]
        colors_in_img = [color for color in colors_in_img if not np.all(color <= 10)]


        return colors_in_img

    except Exception as e:
        print(f"Error: {e}")


# kernel = np.ones((5, 5), np.uint8)

# Function to create color-masked images based on the top colors, excluding black
def create_color_masked_images(image, colors):
    # Load the image
    # image = cv2.imread(image_path)
    image = image
    # Convert the image from BGR to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    images =[]
    # Iterate over the colors
    for i, color in enumerate(colors):
        
        # Check if the color is black (RGB values are [0, 0, 0])
        if not np.all(color <= 10):
            # Create a mask for pixels close to the target color
            lower_bound = np.array(color) - 50
            upper_bound = np.array(color) + 20
            mask = cv2.inRange(image, lower_bound, upper_bound)

            # Apply the mask to the image
            color_masked_image = cv2.bitwise_and(image, image, mask=mask)


            # Morphological closing

            kernel = create_disk_kernel(1)
            color_masked_image = cv2.morphologyEx(color_masked_image, cv2.MORPH_CLOSE, kernel)
            kernel = create_disk_kernel(3)
            color_masked_image = cv2.morphologyEx(color_masked_image, cv2.MORPH_OPEN, kernel)
            color_masked_image = cv2.morphologyEx(color_masked_image, cv2.MORPH_OPEN, kernel)
            color_masked_image = cv2.morphologyEx(color_masked_image, cv2.MORPH_OPEN, kernel)
            color_masked_image = cv2.morphologyEx(color_masked_image, cv2.MORPH_DILATE, kernel)
            color_masked_image = cv2.morphologyEx(color_masked_image, cv2.MORPH_DILATE, kernel)

            # Save the color_masked_image
            # output_path = os.path.join(output_folder, f'color_masked_image_{i}_{os.path.basename(image_path)}')
            # cv2.imwrite(f'aaa{i}.jpg', cv2.cvtColor(color_masked_image, cv2.COLOR_RGB2BGR))
            theOneImage= cv2.cvtColor(color_masked_image, cv2.COLOR_RGB2BGR)
            # gray_image = cv2.cvtColor(theOneImage, cv2.COLOR_RGB2GRAY)

            # cv2.imwrite(f'aaa{i}.jpg',theOneImage)
            images.append(theOneImage)
            # images.append(gray_image)

                
    # print (len(images))
    # print (images)
                  
    return images

def concatenate_three_images(image, output_path):
    ## Read the images
    # img1 = cv2.imread(image1_path)
    # img2 = cv2.imread(image2_path)
    # img3 = cv2.imread(image3_path)

    ## Ensure all images have the same height
    # min_height = min(img1.shape[0], img2.shape[0], img3.shape[0])
    # img1 = img1[:min_height, :]
    # img2 = img2[:min_height, :]
    # img3 = img3[:min_height, :]

    # Concatenate images horizontally
    # combined_img = np.concatenate([img1, img2, img3], axis=1)
    if len(image) :
        combined_img = np.concatenate([image, image, image], axis=1)

        # Save the result
        # cv2.imwrite(output_path, combined_img)
        return combined_img
    else :
        return 




def image_to_base64(image):
        # Read the image
        # image = cv2.imread(image_path)
        
        if image is not None:
            # Convert image to Base64
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode()
            
            # Add the prefix '/9j/4AAQSkZJRgABAQAAAQABAAD/'
            image_base64 = image_base64
            
            return image_base64
        else:
            print("Error: Unable to load the image.")
            return None

async def sendToSolver(base64data):
    # Define the endpoint URL
    # url = "https://api.capsolver.com/createTask"
    url = "http://127.0.0.1:5655/123"

    # # Define the API key
    # api_key = "CAP-1023B2D2D2200C82A98E9FEDC28BF374"
    # api_key = "CAP-1023B2D2D2"        

    # Define the JSON data to be sent in the request
    # json_data = {
    #     "clientKey": api_key,
    #     "task": {
    #         "type": "ImageToTextTask",
    #         "module": "common",
    #         "body": base64data # Base64 encoded image data
    #     }
    # }
    json_data = {
        "base64_image": base64data # Base64 encoded image data

    }
    headers = {
        # "Content-Type": "application/json"
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=json_data, headers=headers) as response:
            if response.status == 200:
                print("Request was successful!")
                print(await response.text())
                return await response.json()
            else:
                print("Error:", response.status)

def compare_first_row(original_image, modified_image):
    original_first_row = original_image[0:1, :100]
    modified_first_row = modified_image[0:1, :100]
    # Calculate similarity metric (e.g., mean squared error)
    similarity_score = cv2.matchTemplate(original_first_row, modified_first_row, cv2.TM_SQDIFF)
    return similarity_score

def find_matching_background(modified_image, original_captchas):
    best_match_index = None
    best_match = None
    min_score = float('inf')
    for i, original_captcha in enumerate(original_captchas):
        similarity_score = compare_first_row(original_captcha, modified_image)
        if similarity_score < min_score:
            min_score = similarity_score
            best_match_index = i
            best_match = original_captcha
    return best_match_index, best_match

def sortTheArrayOfTheImage(arr):
    unsortedarray=[]
    for image in arr:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded_image = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contoursAreas =[]
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            ((x1,y1),(x2,y2),teta)=rect
            area = x2*y2
            contoursAreas.append([area,x1])
        max_area = 0
        max_x = 0
        for area, x in contoursAreas:
            if area > max_area:
                max_area = area
                max_x = x
        unsortedarray.append([image,max_x])
    sorted_contoursAreas = sorted(unsortedarray, key=lambda x: x[1], reverse=False)
    # print([item[1] for item in sorted_contoursAreas])
    # [item[1] for item in sorted_contoursAreas]
    # Return all images from the sorted list
    if sorted_contoursAreas[0][1] == 0:
         sorted_contoursAreas.pop(0)
    # print([item[1] for item in sorted_contoursAreas])
    return [item[0] for item in sorted_contoursAreas]


def solvethesign(image):
    # image = cv2.imread(image_path)


    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a threshold to convert pixels with values less than (10, 10, 10) to black
    _, thresholded_image = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    contoursAreas =[]
    # Draw minimum area rotated bounding boxes around the contours
    for contour in contours:
        # Calculate the minimum area bounding rectangle
        rect = cv2.minAreaRect(contour)
        # print(rect)
        ((x1,y1),(x2,y2),teta)=rect
        area = x2*y2
        contoursAreas.append(area)

        # Get the rotation angle from the rectangle
        angle = rect[2]

        # Limit the angle of rotation to the range -15 to +15 degrees
        if angle > 17:
            angle -= 90
        elif angle < -17:
            angle += 90
        # if -18 <= angle <= 18:
        #     print("hi *-*-*-*-*-*-*-*-*-")
        if -62 <= angle <= -28 or 28 <= angle <= 62:
            # print("hi ++++++++++++++++++")
            return '+'

        # # Rotate the image
        # rows, cols = image.shape[:2]
        # rotation_matrix = cv2.getRotationMatrix2D(rect[0], 0, 1)
        # rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

        # # Rotate the contour points
        # contour_rotated = cv2.boxPoints(rect)
        # contour_rotated = np.intp(contour_rotated)
        # contour_rotated = cv2.transform(np.array([contour_rotated]), rotation_matrix)[0]

        # # Draw the rotated contour
        # cv2.drawContours(rotated_image, [contour_rotated], 0, (0, 255, 0), 2)
    
    if len(contoursAreas):
        mx=max(contoursAreas)
        if mx <8500 :
            return '-'
        else :
            return 'x'
    
    print("we  could not find the math operation im sorry ")

    
    return '-'



# def preprocess_image(image):
#     start_time = time.time()
#     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
#     image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
#     image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)  # Reshape for the model
#     image = (image / 127.5) - 1  # Normalize the image
#     preprocess_time = time.time() - start_time
#     return image, preprocess_time


# def predict(image):
#     start_time = time.time()
#     interpreter.set_tensor(input_details[0]['index'], image)
#     interpreter.invoke()
#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     index = np.argmax(output_data)
#     class_name = class_names[index]
#     confidence_score = output_data[0][index]
#     prediction_time = time.time() - start_time
#     return class_name[2:], confidence_score, prediction_time

# //////////////////////////////////////
# //////////////////////////////////////

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# functions ends 
# Load the modified captcha image
modified_captcha = cv2.imread('all/13.jpg')

original_captchas = []
for i in range(1, 14):
    original_captcha = cv2.imread(f'originall/o{i}.png')
    original_captchas.append(original_captcha)
    
sign=''

# @app.route('/123')
# async def say_hi():
    
#     # image = cv2.imread('./all/11.jpg')
#     image = cv2.imread('./45.png')
#     modified_captcha =image
#     # Find the matching background
#     best_match_index, matching_background = find_matching_background(modified_captcha, original_captchas)
#     # print(f"The best matching background image is original{best_match_index+1}.png")
#     # best_match_filename = f'originall/o{best_match_index + 1}.png'

#     final=imagePreProcess(matching_background,image)
#     colors = get_top_colors(final, top_colors=5)
#     # print(colors)
#     arrayOfImages=create_color_masked_images(final, colors)
#     arrayOfImages=sortTheArrayOfTheImage(arrayOfImages)
#     if len(arrayOfImages)==0:
#         return"errorr zero charecter"
#     if len(arrayOfImages)==1:
#         return"errorr one charecter"
#     if len(arrayOfImages)==2:
#         sign='-'

#     tribleImages =[]
#     for i, image in enumerate(arrayOfImages):
#         filename = f'aaaa{i + 1}.jpg'  # Generate filename dynamically
#         tribleImages.append(
#             concatenate_three_images(image, filename)
#         )
#     first_item = tribleImages[0]
#     last_item = tribleImages[-1]
#     combined_img = np.concatenate([first_item, last_item], axis=1)
#     # cv2.imwrite("./combined_img.jpg",combined_img)


#     solve = []
#     tasks = []

#     base64Data =image_to_base64(combined_img)
#     # print(base64)
#     task = asyncio.create_task(sendToSolver(base64Data))
#     tasks.append(task)
#     # print(task)
#     if len(arrayOfImages)>=3:
#         arrayOfImages[1]
#         sign=solvethesign(arrayOfImages[1])

#     solve = await asyncio.gather(*tasks)
#     solve.append(sign)
    
#     # print(solve)

#     return solve

prefix1 = 'data:image/jpeg;base64,'
prefix2 = 'data:image/jpg;base64,'
sign=''
@app.route('/image' ,methods=['POST'])
async def recive_theImage():
    try:
        json_data = request.get_json()

        if 'base64_image' in json_data:
            base64_image = json_data['base64_image']
            if base64_image.startswith(prefix1):
                base64_image = base64_image[len(prefix1):]
            if base64_image.startswith(prefix2):
                base64_image = base64_image[len(prefix2):]
 
            # Decode the base64 image
            image_data = base64.b64decode(base64_image)
            np_array = np.frombuffer(image_data, np.uint8)
            # cv2.imwrite("loo.jpg",image_data)
            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            # cv2.imwrite('45.png',image)
                        
            modified_captcha =image
            # Find the matching background
            best_match_index, matching_background = find_matching_background(modified_captcha, original_captchas)
            # print(f"The best matching background image is original{best_match_index+1}.png")
            # best_match_filename = f'originall/o{best_match_index + 1}.png'

            final=imagePreProcess(matching_background,image)
            cv2.imwrite('./AAAAAAAAfinal_output_path.jpg', final) 
            # colors = get_top_colors(final, top_colors=5)
            # # print(colors)
            # arrayOfImages=create_color_masked_images(final, colors)
            # arrayOfImages=sortTheArrayOfTheImage(arrayOfImages)
            # if len(arrayOfImages)==0:
            #     return"errorr zero charecter"
            # if len(arrayOfImages)==1:
            #     return"errorr one charecter"
            # if len(arrayOfImages)==2:
            #     sign='-'
            arrayOfImages = main(final)

            tribleImages =[]
            solvetasks = []
            tasks = []
            for i, image in enumerate(arrayOfImages):
                
                # imageProcessed, timed = preprocess_image(image)
                
                tribleImages.append(
                    image
                )
                base64Data =image_to_base64(image)
                solvetasks.append(await sendToSolver(base64Data))
                task = asyncio.create_task(sendToSolver(base64Data))
                tasks.append(task)

            # first_item = tribleImages[0]
            # second_item = tribleImages[1]
            # last_item = tribleImages[-1]
            # combined_img = np.concatenate([first_item, last_item], axis=1)


            # base64Data =image_to_base64(combined_img)
            # print(base64)
            # task = asyncio.create_task(sendToSolver(base64Data))
            # task = asyncio.create_task(predict())
            # tasks.append(task)
            # print(task)
            # if len(arrayOfImages)>=3:
            #     arrayOfImages[1]
            #     sign=solvethesign(arrayOfImages[1])
            solvetasks = await asyncio.gather(*tasks)
            # solvetasks.append(sign)
            solve ={'solvetasks':solvetasks}
            
            # print(solve)
            return solve

        return 'Invalid JSON data'

    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,threaded=True)

    # http_server = WSGIServer(('127.0.0.1', 5000), app, spawn='eventlet')
    # http_server.serve_forever()