from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
# from sklearn.cluster import KMeans
# import os
from sklearn.cluster import MiniBatchKMeans
import requests

app = Flask(__name__)

# functions starts

def create_disk_kernel(radius):

    # Ensure the radius is a non-negative integer
    radius = max(0, int(radius))

    # Create a grid of coordinates
    x, y = np.ogrid[-radius:radius+1, -radius:radius+1]

    # Create a binary matrix representing a disk
    disk_matrix = x**2 + y**2 <= radius**2

    return disk_matrix.astype(np.uint8)


def imagePreProcess(theOriginalPath,image):

    image_o = cv2.imread(theOriginalPath)
    image_path ='./light/1+8.jpg'
    # image_e = cv2.imread(image_path)
    image_e = image

    result = cv2.subtract(image_o, image_e)
    result2 = cv2.subtract(image_e, image_o)

    add = cv2.add(result, result2)
    gray_image = cv2.cvtColor(add, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)

    kernel = create_disk_kernel(2)
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
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=2)

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

    # cv2.imwrite('AAAAAAAAAAAAAfinal_output_path.jpg', final)    
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
        print(colors_in_img)

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
            lower_bound = np.array(color) - 20
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






def sendToSolver(base64):
    # Define the endpoint URL
    url = "https://api.capsolver.com/createTask"

    # Define the API key
    # api_key = "CAP-1023B2D2D2200C82A98E9FEDC28BF374"
    api_key = "CAP-1023B2D2D2"

    # Define the JSON data to be sent in the request
    json_data = {
        "clientKey": api_key,
        "task": {
            "type": "ImageToTextTask",
            "module": "common",
            "body": base64 # Base64 encoded image data
        }
    }
    headers = {
        "Host": "api.capsolver.com",
        "Content-Type": "application/json"
    }

    # Send the POST request
    response = requests.post(url, json=json_data,headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        print("Request was successful!")
        print("Response:")
        print(response.json())
        return response.json()
    else:
        print("Error:", response.status_code)




# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# functions ends 
    

@app.route('/123')
def say_hi():
    
    image = cv2.imread('./8+1.jpg')
    final=imagePreProcess('Original.png',image)
    colors = get_top_colors(final, top_colors=5)
    print(colors)
    arrayOfImages=create_color_masked_images(final, colors)
    tribleImages =[]
    for i, image in enumerate(arrayOfImages):
        filename = f'aaaa{i + 1}.jpg'  # Generate filename dynamically
        tribleImages.append(
            concatenate_three_images(image, filename)
        )
    solve=[]
    for i, image in enumerate(tribleImages):
        base64 =image_to_base64(image)
        # print(base64)
        result =sendToSolver(base64)
        solve.append(result)
        print(solve)

        
    
    
    

    return solve

@app.route('/image' ,methods=['POST'])
def recive_theImage():
    try:
        json_data = request.get_json()

        if 'base64_image' in json_data:
            base64_image = json_data['base64_image']
 
            # Decode the base64 image
            # image_data = base64.b64decode(base64_image)

            # # Here you can process the image data as needed
            # For example, you can save it to a file, perform image processing, etc.

            return 'Image uploaded and processed successfully'

        return 'Invalid JSON data'

    except Exception as e:
        return f'Error: {str(e)}'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
