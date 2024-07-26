import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def load_image(image_or_path):
    if isinstance(image_or_path, str):
        # Assume it is a path and try to load the image
        image = cv2.imread(image_or_path)
        if image is None:
            raise ValueError(f"Failed to load image from path: {image_or_path}")
        return image
    elif isinstance(image_or_path, np.ndarray):
        # Assume it is already an image
        return image_or_path
    else:
        raise ValueError("Invalid input: input must be a file path string or a numpy array representing an image.")

def preprocess_image(image):
    """Convert the image to grayscale and apply GaussianBlur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def apply_adaptive_threshold(blurred):
    """Apply adaptive thresholding to the blurred image to create a binary image."""
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return binary

def apply_morphological_gradient(binary):
    """Apply morphological gradient to the binary image to highlight edges."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    return morph_gradient

def find_and_sort_contours(morph_gradient):
    """Find contours in the morphologically processed image and sort them from left to right."""
    
    contours, _ = cv2.findContours(morph_gradient, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
    return sorted_contours

def draw_contours(image, contours):
    """Draw contours on the image."""
    contoured_image = image.copy()
    cv2.drawContours(contoured_image, contours, -1, (0, 255, 0), 2)
    return contoured_image

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

        # Remove colors similar to black or less than [10, 10, 10]
        colors_in_img = [color for color in colors_in_img if not np.all(color <= 10)]

        return colors_in_img

    except Exception as e:
        print(f"Error: {e}")

def create_disk_kernel(radius):

    # Ensure the radius is a non-negative integer
    radius = max(0, int(radius))

    # Create a grid of coordinates
    x, y = np.ogrid[-radius:radius+1, -radius:radius+1]

    # Create a binary matrix representing a disk
    disk_matrix = x**2 + y**2 <= radius**2

    return disk_matrix.astype(np.uint8)


# def create_color_masked_images(image, colors):
#     # Convert the image from BGR to RGB format
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     images = []
#     # Iterate over the colors
#     for i, color in enumerate(colors):
#         # Check if the color is black (RGB values are [0, 0, 0])
#         if not np.all(color <= 10):
#             # Create a mask for pixels close to the target color
#             lower_bound = np.array(color) - 50
#             upper_bound = np.array(color) + 40
#             mask = cv2.inRange(image, lower_bound, upper_bound)

#             # Apply the mask to the image
#             color_masked_image = cv2.bitwise_and(image, image, mask=mask)

#             # Morphological closing
#             kernel = create_disk_kernel(1)
#             color_masked_image = cv2.morphologyEx(color_masked_image, cv2.MORPH_CLOSE, kernel)
#             kernel = create_disk_kernel(3)
#             color_masked_image = cv2.morphologyEx(color_masked_image, cv2.MORPH_OPEN, kernel)
#             color_masked_image = cv2.morphologyEx(color_masked_image, cv2.MORPH_OPEN, kernel)
#             color_masked_image = cv2.morphologyEx(color_masked_image, cv2.MORPH_OPEN, kernel)
#             color_masked_image = cv2.morphologyEx(color_masked_image, cv2.MORPH_DILATE, kernel)
#             # color_masked_image = cv2.morphologyEx(color_masked_image, cv2.MORPH_DILATE, kernel)

#             theOneImage = cv2.cvtColor(color_masked_image, cv2.COLOR_RGB2BGR)
#             images.append(theOneImage)

#     return images

def create_color_masked_images(image, colors, kernel_size_close=1, kernel_size_open=3, kernel_size_dilate=3):
    # Convert the image from BGR to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    images = []
    # Iterate over the colors
    for i, color in enumerate(colors):
        # Check if the color is black (RGB values are [0, 0, 0])
        if not np.all(color <= 10):
            # Ensure color bounds do not exceed the valid range (0-255)
            lower_bound = np.clip(np.array(color) - 50, 0, 255)
            upper_bound = np.clip(np.array(color) + 40, 0, 255)
            mask = cv2.inRange(image, lower_bound, upper_bound)

            # Apply the mask to the image
            color_masked_image = cv2.bitwise_and(image, image, mask=mask)

            # Morphological operations
            kernel_close = create_disk_kernel(kernel_size_close)
            kernel_open = create_disk_kernel(kernel_size_open)
            kernel_dilate = create_disk_kernel(kernel_size_dilate)

            color_masked_image = cv2.morphologyEx(color_masked_image, cv2.MORPH_CLOSE, kernel_close)
            color_masked_image = cv2.morphologyEx(color_masked_image, cv2.MORPH_OPEN, kernel_open)
            color_masked_image = cv2.morphologyEx(color_masked_image, cv2.MORPH_DILATE, kernel_dilate)

            theOneImage = cv2.cvtColor(color_masked_image, cv2.COLOR_RGB2BGR)
            images.append(theOneImage)
        else:
            print(f"Skipping near-black color: {color}")

    return images


def find_and_sort_contours_normal_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[2]*cv2.boundingRect(contour)[3])
    return contours

def get_min_x(contours):
    if contours:
        return min(cv2.boundingRect(contour)[0] for contour in contours)
    return float('inf')

def dilate_color_image(image, kernel_size=3, iterations=1):
    # Split the image into its color channels
    b_channel, g_channel, r_channel = cv2.split(image)

    # Create the dilation kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Apply dilation to each channel separately
    b_dilated = cv2.dilate(b_channel, kernel, iterations=iterations)
    g_dilated = cv2.dilate(g_channel, kernel, iterations=iterations)
    r_dilated = cv2.dilate(r_channel, kernel, iterations=iterations)

    # Merge the dilated channels back into a single image
    dilated_image = cv2.merge((b_dilated, g_dilated, r_dilated))

    return dilated_image

def formate_the_image_to_cropped_image_and_whit_bgcolor(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_image = cv2.bitwise_not(gray_image)

    # Threshold the image to create a binary image
    _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Define padding
    padding = 10

    # Adjust the bounding box with padding
    x_padded = max(x - padding, 0)
    y_padded = max(y - padding, 0)
    w_padded = min(w + 2 * padding, gray_image.shape[1] - x_padded)
    h_padded = min(h + 2 * padding, gray_image.shape[0] - y_padded)

    # Crop the image to the padded bounding box
    cropped_image = gray_image[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
    # cv2.imwrite('asdasacr.jpg',cropped_image)
    return cropped_image

def contours_process(contours,image):
    color_images = []

    if len(contours) == 1:
        print("1")
        colors = get_top_colors(image, 4)
        print("Top colors:", colors)
        if len(colors)<3:
            colors =get_top_colors(image, 4,1)

        print("Top colors:", colors)
        color_images = create_color_masked_images(image, colors)
        if len(color_images) < 3:
            colors = get_top_colors(image, 5)
            color_images = create_color_masked_images(image, colors)
            print("Not enough color images generated.")
            return
        # find countors
        contour0, y0, w0, h0=cv2.boundingRect(find_and_sort_contours_normal_image(color_images[0])[0])
        contour1, y1, w1, h1=cv2.boundingRect(find_and_sort_contours_normal_image(color_images[1])[0])
        contour2, y2, w2, h2=cv2.boundingRect(find_and_sort_contours_normal_image(color_images[2])[0])
        
        color_images_with_contours = [
            (color_images[0], contour0),
            (color_images[1], contour1),
            (color_images[2], contour2)
            ]
        color_images_with_contours.sort(key=lambda item: item[1])
        color_images_with_contours =[image for image, x in color_images_with_contours]
        
        color_images_with_contours[0]=image[y0:y0+h0, contour0:contour0+w0]
        color_images_with_contours[1]=image[y1:y1+h1, contour1:contour1+w1]
        color_images_with_contours[2]=image[y2:y2+h2, contour2:contour2+w2]

        color_images = color_images_with_contours
        # color_images.append([color_images[0],contour0])
        # color_images.append([color_images[1],contour1])
        # color_images.append([color_images[2],contour2])
        # sort contours
        
        # Ensure 3 images
        # while len(color_images) < 3:
    
    elif len(contours) == 2:
        print("There are 2 of them, baba")
        largest_contour = contours[0]  # The largest contour
        smallest_contour = contours[1]  # The smallest contour

        # Extracting largest and smallest contour images
        x1, y, w, h = cv2.boundingRect(largest_contour)
        area1 = w * h
        largest_character = image[y:y+h, x1:x1+w]
        x2, y, w, h = cv2.boundingRect(smallest_contour)
        area2 = w * h
        smallest_character = image[y:y+h, x2:x2+w]
        if area1<area2 :
            largest_character, smallest_character = smallest_character, largest_character

        # Get top colors from the largest contour
        colors_largest = get_top_colors(largest_character, 3)
        print("Top colors:", colors_largest)
        if len(colors_largest)<2:
            colors_largest =get_top_colors(image, 3,1)
            print("Top colors:", colors_largest)
        if len(colors_largest)<2:
            colors_largest =get_top_colors(image, 4,1)
            print("Top colors:", colors_largest)

        print("Top colors in the largest contour:", colors_largest)
        largest_color_images = create_color_masked_images(largest_character, colors_largest)
        contour0=find_and_sort_contours_normal_image(largest_color_images[0])
        contour1=find_and_sort_contours_normal_image(largest_color_images[1])
        contour0=contour0[0]
        contour1=contour1[0]
        x3, y, w, h=cv2.boundingRect(contour0)
        x4, y, w, h=cv2.boundingRect(contour1)
        if x4<x3:
            largest_color_images[0],largest_color_images[1]=largest_color_images[1],largest_color_images[0]
        # Get top colors from the smallest contour
        # colors_smallest = get_top_colors(smallest_character, 2)
        # print("Top colors in the smallest contour:", colors_smallest)
        # smallest_color_images = create_color_masked_images(smallest_character, colors_smallest)

        # u should find the contoures here and 
        if x1<x2:
            color_images.append(largest_color_images[0])
            color_images.append(largest_color_images[1])
            color_images.append(smallest_character)
        else :
            color_images.append(smallest_character)
            color_images.append(largest_color_images[0])
            color_images.append(largest_color_images[1])
        # largest_color_images[:2] + smallest_character

    elif len(contours) == 3:
        print("3 characters will be processed")
        for contour in contours[:3]:
            x, y, w, h = cv2.boundingRect(contour)
            character = image[y:y+h, x:x+w]
            # colors = get_top_colors(character, 1)
            # print("Top colors in the character:", colors)
            # color_images += create_color_masked_images(character, colors)
            color_images.append((character,x))
            color_images.sort(key=lambda item: item[1])
    
        # Unpack the sorted images
        img0,x0 = color_images[0]
        img1,x1 = color_images[1]
        img2,x2 = color_images[2]
        color_images =[image for image, x in color_images]

        
        print(f"Sorted x-coordinates: {x0}, {x1}, {x2}")
    else:
        print(f"There are {len(contours)} contours detected")

        image1 = load_image(image)
        kernel=create_disk_kernel(3)
        # image1=dilate_color_image(image1,3,4)
        cv2.imwrite('asd.jpg',image1)
        blurred = preprocess_image(image1)
        binary = apply_adaptive_threshold(blurred)
        binary = cv2.dilate(binary, kernel, iterations=2)
        morph_gradient = apply_morphological_gradient(binary)
        contours = find_and_sort_contours(morph_gradient)
        return contours_process(contours,image1)


    return color_images 



def main(image_path):
    image = load_image(image_path)
    blurred = preprocess_image(image)
    binary = apply_adaptive_threshold(blurred)
    morph_gradient = apply_morphological_gradient(binary)
    contours = find_and_sort_contours(morph_gradient)

    # Draw contours on the original image
    contoured_image = draw_contours(image, contours)
    cv2.imwrite('contoured_image.jpg', contoured_image)
    
    color_images=contours_process(contours,image)




    
    # # Save the images
    # for i, (img,contourses) in enumerate(color_images):
    #     cv2.imwrite(f'single{i}.jpg', img)
    for i, img in enumerate(color_images):
        color_images[i]=formate_the_image_to_cropped_image_and_whit_bgcolor(img)
        cv2.imwrite(f'single{i}.jpg',  color_images[i])

    return color_images

# Path to the image
image_path = "akhra.jpg"
imegsggg=main(image_path)
# cv2.imwrite('single0.jpg',imegsggg[0])
# cv2.imwrite('single1.jpg',imegsggg[1])
# cv2.imwrite('single2.jpg',imegsggg[2])
