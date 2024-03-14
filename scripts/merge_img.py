from PIL import Image

def merge_images_horizontally(image_paths, output_path):
    images = [Image.open(image_path) for image_path in image_paths]

    # Ensure that all images have the same height
    max_height = 512
    images = [img if img.height == max_height else img.resize((img.width * max_height // img.height, max_height)) for img in images]

    # Calculate the total width for the merged image
    total_width = sum(img.width for img in images)

    # Create a blank image with the total width and the maximum height
    merged_image = Image.new('RGB', (total_width, max_height))

    # Paste each image into the merged image
    x_offset = 0
    for img in images:
        merged_image.paste(img, (x_offset, 0))
        x_offset += img.width

    # Save the merged image
    merged_image.save(output_path)

if __name__ == "__main__":
    image_paths_10955 = ["/home/user001/zwl/zyx/CodeFormer-master/inp6/lq/10955.png", 
                   "/home/user001/zwl/zyx/GPEN-main/eval_img_test/10955.png", 
                   "/home/user001/zwl/zyx/CodeFormer-master/inp6/10955.png", 
                   "/home/user001/zwl/zyx/Diffbir/outputs/inp-re3/z2/10955.png", 
                   "/home/user001/zwl/zyx/Diffbir/outputs/inp-re3/hq/10955.png"]
    
    image_paths_1777 = ["/home/user001/zwl/zyx/CodeFormer-master/inp11/lq/1777.png", 
                   "/home/user001/zwl/zyx/GPEN-main/eval_img_test/1777.png", 
                   "/home/user001/zwl/zyx/CodeFormer-master/inp11/1777.png", 
                   "/home/user001/zwl/zyx/Diffbir/outputs/inp-re10/z2/1777.png", 
                   "/home/user001/zwl/zyx/Diffbir/outputs/inp-re10/hq/1777.png"]
    
    output_path = "merged_image_10955.png"

    merge_images_horizontally(image_paths_10955, output_path)
