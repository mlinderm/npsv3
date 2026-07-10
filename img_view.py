import webdataset as wds
from PIL import Image

images_path = "/storage/mlinderman/projects/sv/npsv3-experiments/training/hg002v1.1.dipcall.passing.sv.hg38.images/generator=coverage,pileup=unphased,simulation.replicates=1/images-{0000..0015}.tar"
dataset = wds.WebDataset(images_path, shardshuffle=False).decode()

def find_img(img_idx):
    for _i, image in enumerate(dataset):
        if _i == img_idx:
            return image

img_idx = 6204
img_data = find_img(img_idx)
for _i, sim_img in enumerate(img_data["sim.images.npy.gz"]):
    img = Image.fromarray(sim_img[0][:, :, [0, 1, 5]])
    img.save(f"{img_idx}/{_i}.png")