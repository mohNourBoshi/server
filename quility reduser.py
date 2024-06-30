import cv2

def reduce_image_quality(input_image_path, output_image_path, quality):
    # Read the image
    image = cv2.imread(input_image_path)
    
    # Set JPEG quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    
    # Save the image with the specified quality
    cv2.imwrite(output_image_path, image, encode_param)
    print(f"Image saved at {output_image_path} with quality {quality}")

# Example usage
input_image_path = './aaa0.jpg'
output_image_path = './p.jpg'
quality = 10  # Quality value (0 to 100), lower means more compression and lower quality

reduce_image_quality(input_image_path, output_image_path, quality)
