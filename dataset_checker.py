import webdataset as wds
from PIL import Image
import sys

def save_img(img, img_idx):
    img.save(f"ada_output/{job_num}/images/output_image_{img_idx}.png")

job_num = sys.argv[1]
# images_path = "/storage/mlinderman/projects/sv/npsv3-experiments/training/hg002v1.1.dipcall.passing.sv.hg38.images/generator=coverage,pileup=unphased,simulation.replicates=1/images-{0000..0015}.tar"
images_path = "/storage/mlinderman/projects/sv/npsv3-experiments/training/hgsvc3-hprc-2024-02-23.dipcall.passing.hg38.eval-images/HG00733/generator=coverage,pileup=unphased_variable,simulation.replicates=1/images-{0000..0015}.tar"
dataset = wds.WebDataset(images_path, shardshuffle=False).decode()
real_count = {}
for _i, sample in enumerate(dataset):
    real_label = sample["label.cls"]
    image = Image.fromarray(sample["image.npy.gz"][:, :, [0, 1, 5]])
    width, height = image.size
    save_img(image, _i)
    print(_i, width, height)
    
    real_count.setdefault(real_label, 0)
    real_count[real_label] += 1
    if _i >= 100:
        break
print(f"num samples: {_i+1}")
print(real_count)
# print(region)