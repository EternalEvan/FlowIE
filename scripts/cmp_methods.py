from PIL import Image
import os
import pdb
def merge_images_with_same_name(folder_paths, output_folder):
    # Get the list of filenames in the first folder
    os.makedirs(output_folder,exist_ok=True)
    filenames = sorted(os.listdir(folder_paths[0]))
    num = 0
    for filename in filenames:
        images = []
     
        #if filename != '00000530.png':
        #    continue
        # Load images with the same name from each folder
        for i,folder_path in enumerate(folder_paths):
            file_path = os.path.join(folder_path, filename)
            #pdb.set_trace()
            if os.path.isfile(file_path):
                image = Image.open(file_path)
                if i == 0:
                    w,h = image.size
                else:
                    image = image.resize((w,h))
                images.append(image)
            else:
                print(file_path)
                continue

        # Create a new blank image to merge the others into
        total_width = sum(image.width for image in images)
        max_height = max(image.height for image in images)
        merged_image = Image.new("RGB", (total_width, max_height))

        # Paste each image into the merged image
        current_width = 0
        for image in images:
            merged_image.paste(image, (current_width, 0))
            current_width += image.width

        # Save the merged image
        output_path = os.path.join(output_folder, f"{filename}")
        merged_image.save(output_path)
        num += 1
        print(num)

if __name__ == "__main__":
    # Provide the paths to the four folders
    celeba_test_folders = [
        "/data1/zyx/celeba_512_validation_lq",
        #"/home/zyx/VQFR/celeba-test-0.5/restored_faces",
        "/home/zyx/GPEN-main/celeba-test-results",
        "/home/zyx/RestoreFormer/celeba-test-results/restored_faces",
        "/home/zyx/GFPGAN-master/celeba-test-results/restored_faces",
        "/home/zyx/DMDNet/celeba-test-results",
        "/home/zyx/DiffBIR-main/outputs/celebatest_diffbir_512",
        "/home/zyx/DiffBIR-main/outputs/celebatest_reflow_249999/midd",
        "/data1/zyx/celeba_512_validation"
    ]

    lfw_test_folders = [
        "/data1/zyx/Lfw-Test",
        #"/home/zyx/VQFR/celeba-test-0.5/restored_faces",
        "/home/zyx/GPEN-main/lfw-test-results",
        "/home/zyx/RestoreFormer/lfw-test-results/restored_faces",
        "/home/zyx/GFPGAN-master/lfw-test-results/restored_faces",
        "/home/zyx/DMDNet/lfw-test-results",
        "/home/zyx/CodeFormer-master/lfw-test-results/restored_faces",
        "/home/zyx/DiffBIR-main/outputs/diffbir_lfw",
        "/home/zyx/DiffBIR-main/outputs/reflow_lfw/midd",
     
    ]   
    wider_test_folders = [
        "/data1/zyx/Wider-Test",
        #"/home/zyx/VQFR/celeba-test-0.5/restored_faces",
        "/home/zyx/GPEN-main/wider-test-results",
        "/home/zyx/RestoreFormer/wider-test-results/restored_faces",
        "/home/zyx/GFPGAN-master/wider-test-results/restored_faces",
        "/home/zyx/DMDNet/wider-test-results",
        "/home/zyx/CodeFormer-master/wider-test-results/restored_faces",
        "/home/zyx/DiffBIR-main/outputs/diffbir_wider",
        "/home/zyx/DiffBIR-main/outputs/reflow_wider/midd",
     
    ] 

    child_test_folders = [
        "/data1/zyx/CelebChild-Test/Child",
        #"/home/zyx/VQFR/celeba-test-0.5/restored_faces",
        "/home/zyx/GPEN-main/celebchild-test-results",
        "/home/zyx/RestoreFormer/celebchild-test-results/restored_faces",
        "/home/zyx/GFPGAN-master/celebchild-test-results/restored_faces",
        "/home/zyx/DMDNet/celebchild-test-results",
        "/home/zyx/CodeFormer-master/celebchild-test-results/restored_faces",
        "/home/zyx/DiffBIR-main/outputs/celeba_child_diffbir",
        "/home/zyx/DiffBIR-main/outputs/reflow_child/midd",
     
    ] 
    
    inpainting_folders = [
        "/home/user001/zwl/zyx/CodeFormer-master/inp11/lq",
        "/home/user001/zwl/zyx/CodeFormer-master/inp11",
        "/home/user001/zwl/zyx/Diffbir/outputs/inp-re10/z2",
        "/home/user001/zwl/zyx/Diffbir/outputs/inp-re10/hq"
        
    ]
    
    teaser_folders = [
        "/home/user001/zwl/zyx/Diffbir/outputs/inp-re7/lq",
       
        "/home/user001/zwl/zyx/Diffbir/outputs/inp-re7/z2",
       "/home/user001/zwl/zyx/Diffbir/outputs/inp-re7/hq"
        
    ]
    # Specify the output folder for the merged images
    
    celeba_lqlq_test_folders = [
        "/home/user001/zwl/data/celeba_512_validation_lq_lq",
        #"/home/zyx/VQFR/celeba-test-0.5/restored_faces",
        "/home/user001/zwl/zyx/GPEN-main/lqlq_results",
        "/home/user001/zwl/zyx/RestoreFormer-main/lqlq_results/test/restored_faces",
        "/home/user001/zwl/zyx/GFPGAN-master/lqlq_results/restored_faces",
        "/home/user001/zwl/zyx/GPEN-main/DMDNet/lqlq_results",
        "/home/user001/zwl/zyx/Diffbir/outputs/diffbir_celeba_lq_than_lq",
        "/home/user001/zwl/zyx/Diffbir/outputs/midd",
        "/home/user001/zwl/data/celeba_512_validation"
    ]
    derain_folders = ["/home/user001/zwl/data/Derain/Rain100L/rainy",
        "/home/user001/zwl/zyx/PReNet/results/Rain100L/PReNet",
        "/home/user001/zwl/zyx/RCDNet-master/RCDNet_code/for_syn/experiment/RCDNet_test/results",
        #"/home/user001/zwl/zyx/Pretrained-IPT/experiment/results/ipt/results-DIV2K",
        "/home/user001/zwl/zyx/Diffbir/outputs/swin_derain",
    ]
    rainy_folders = [#"/home/user001/zwl/data/Derain/Rain100L/",
        "/home/user001/zwl/zyx/PReNet/results/Rain100L/PReNet/rainy",
        #"/home/user001/zwl/zyx/Diffbir/outputs/cmp_derain/rcd",
        "/home/user001/zwl/zyx/RCDNet-master/RCDNet_code/for_syn/experiment/RCDNet_test/results/rainy",
        "/home/user001/zwl/zyx/Diffbir/outputs/swin_derain/rainy",
    ]
    output_folder = "outputs/cmp_derain/"
    

    merge_images_with_same_name(derain_folders, output_folder)
