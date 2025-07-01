import os
import shutil
from pathlib import Path

def create_voc_structure(dataset_name="VOCdevkit"):
    # Create VOC directory structure
    base_path = f"./data/{dataset_name}"
    paths = [
        f"{base_path}/VOC2007/Annotations",
        f"{base_path}/VOC2007/ImageSets/Main",
        f"{base_path}/VOC2007/JPEGImages",
        f"{base_path}/test/VOC2007/Annotations",
        f"{base_path}/test/VOC2007/ImageSets/Main",
        f"{base_path}/test/VOC2007/JPEGImages"
    ]
    
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
    
    return base_path

def copy_files(src_path, annotation_dest, images_dest):
    # Get all XML files
    files = [f for f in os.listdir(src_path) if f.endswith('.xml') or f.endswith('.jpg')]
    file_bases = set()

    for f in files:
        base_name = f.rsplit('.', 1)[0]
        if base_name not in file_bases:
            file_bases.add(base_name)
            
            # Copy XML file
            xml_src = os.path.join(src_path, f"{base_name}.xml")
            if os.path.exists(xml_src):
                shutil.copy2(xml_src, os.path.join(annotation_dest, f"{base_name}.xml"))
            
            # Copy image file
            img_src = os.path.join(src_path, f"{base_name}.jpg")
            if os.path.exists(img_src):
                shutil.copy2(img_src, os.path.join(images_dest, f"{base_name}.jpg"))
    
    return list(file_bases)

def create_image_sets(file_bases, output_path, filename):
    with open(os.path.join(output_path, filename), 'w') as f:
        f.write('\n'.join(file_bases))

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert RDD dataset to VOC format')
    parser.add_argument('--dataset_name', default='VOCdevkit', help='Name of the output dataset folder')
    args = parser.parse_args()

    # Source paths for rdd11k
    rdd_train_path = "./rdd11k/train"
    rdd_val_path = "./rdd11k/valid"  
    rdd_test_path = "./rdd11k/test"

    # Create VOC structure
    voc_base = create_voc_structure(args.dataset_name)
    
    # Copy training and validation files
    train_files = copy_files(rdd_train_path, 
                           f"{voc_base}/VOC2007/Annotations",
                           f"{voc_base}/VOC2007/JPEGImages")
    
    val_files = copy_files(rdd_val_path,
                          f"{voc_base}/VOC2007/Annotations",
                          f"{voc_base}/VOC2007/JPEGImages")
    
    # Copy test files
    test_files = copy_files(rdd_test_path,
                           f"{voc_base}/test/VOC2007/Annotations",
                           f"{voc_base}/test/VOC2007/JPEGImages")
    
    # Create image set files
    create_image_sets(train_files + val_files, 
                     f"{voc_base}/VOC2007/ImageSets/Main",
                     "trainval.txt")
    
    create_image_sets(test_files,
                     f"{voc_base}/test/VOC2007/ImageSets/Main",
                     "test.txt")
    
    print("\nConversion completed!")
    print(f"Training + Validation images: {len(train_files) + len(val_files)}")
    print(f"Test images: {len(test_files)}")
    print(f"\nTo train the model, use this command:")
    print(f"python train_ssd_VOC.py --datasets ./data/{args.dataset_name}/VOC2007 --validation_dataset ./data/{args.dataset_name}/test/VOC2007 --net mb2-ssd-lite --batch_size 32 --num_epochs 200")

if __name__ == "__main__":
    main()