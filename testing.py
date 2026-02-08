import os

image_folder = './val/processed_images/'
gt_folder = './val/gt_txt/'

images = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
texts = sorted([f for f in os.listdir(gt_folder) if f.endswith('.txt')])

print(f"Total images: {len(images)}")
print(f"Total texts: {len(texts)}")
print(f"\nFirst 10 images:")
for img in images[:10]:
    print(f"  {img}")
print(f"\nFirst 10 texts:")
for txt in texts[:10]:
    print(f"  {txt}")