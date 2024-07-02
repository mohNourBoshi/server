import cv2
from extarcting_the_charecters import main

# Path to the image you want to process
image_path = "AAAAAAAAfinal_output_path.jpg"

# Call the main function to process the image
result = main(image_path)

# Display each image in the result using cv2.imshow()
if len(result) >= 3:  # Ensure at least three images are returned
    cv2.imwrite("Image 1.jpg", result[0])
    cv2.imwrite("Image 2.jpg", result[1])
    cv2.imwrite("Image 3.jpg", result[2])

else:
    print("Less than 3 images returned from main function.")
    print("Result:", result)
