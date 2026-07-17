import webdataset as wds
from PIL import Image
from IPython.display import display
import sys
import os
import time
import random
import numpy as np

# Torch ecosystem
import torch
import torchvision
import torch.nn.functional as F

from transformers import AutoImageProcessor, AutoModel, pipeline
import pandas as pd

from huggingface_hub import login

# TODO: Figure out how to get this token as an environmental variable
token = os.getenv("HF_TOKEN")
login(token=token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def prep_genomic_image(data, resize):
    # use three channels
    img_array = data[:, :, [0, 1, 5]]
    img = Image.fromarray(img_array).convert("RGB")
    # print(img.size)
    return processor(do_resize=resize, images=img, return_tensors="pt").to(device)

def predict_sample_label(sample, model, comparison_metric):
    doResize = False
    # Predict for a 'Real' Sample
    real_data = sample["image.npy.gz"]
    real_label = sample["label.cls"]
    real_inputs = prep_genomic_image(real_data, doResize)
    correct_sim_input = prep_genomic_image(sample["sim.images.npy.gz"][real_label][0], doResize)

    with torch.no_grad():
        real_emb = model(**real_inputs).pooler_output
        norm_real_emb = F.normalize(real_emb, p=2, dim=1)
        correct_sim_emb = model(**correct_sim_input).pooler_output

    correct_dist = torch.cdist(real_emb, correct_sim_emb, 2.0).squeeze(0)[0].item()

    sim_embs = []
    euclid_dists = []
    similarities = []

    # Loop through associated sim images and get embeddings
    for _i, sim_sample in enumerate(sample["sim.images.npy.gz"]):
        sim_data = sim_sample[0]
        # display(Image.fromarray(sim_data[:, :, [0, 1, 5]]))

        # Extract DINO embedding
        inputs = prep_genomic_image(sim_data, doResize)
        # print(inputs.pixel_values.shape)
        with torch.no_grad():
            sim_emb = model(**inputs).pooler_output
            sim_embs.append(sim_emb)
            euclid_dist = torch.cdist(real_emb, sim_emb, 2.0).squeeze(0)[0].item()

            sim_emb = F.normalize(sim_emb, p=2, dim=1) # Normalize
            similarity = torch.mm(norm_real_emb, sim_emb.T).item()

        similarities.append(similarity)
        euclid_dists.append(euclid_dist)

    # Compares embeddings with euclidean distance
    if comparison_metric == "euclid_dist":
        predicted_label = euclid_dists.index(min(euclid_dists))
        img_dist = euclid_dists[predicted_label]
        correct_pred_dist = torch.cdist(correct_sim_emb, sim_embs[predicted_label], 2.0).squeeze(0)[0].item()
        return predicted_label, img_dist, correct_pred_dist, real_inputs

    # Compares embeddigns with cosine similarity
    # I haven't tested this in a while so it might be broken
    if comparison_metric == "cos_sim":
        predicted_label = similarities.index(max(similarities))
        similarity = similarities[predicted_label]
        sim_emb = F.normalize(sim_embs[predicted_label], p=2, dim=1)
        correct_sim_emb = F.normalize(correct_sim_emb, p=2, dim=1)
        correct_pred_dist = torch.mm(norm_real_emb, sim_emb.T).item()
        return predicted_label, similarity, correct_pred_dist, real_inputs

# Uses predicted presence or absence of a variant to determine correctness
def presence_absence(real_label, pred_label, confusion_matrix):
    correct = 0
    if real_label == 0:
        if pred_label == 0:
            # predicted zero and was zero
            confusion_matrix[0][0] += 1
            correct = 1
        else:
            # predicted one and was zero
            confusion_matrix[1][0] += 1
    elif real_label >= 1:
        if pred_label >= 1:
            # predicted one and was one
            confusion_matrix[1][1] += 1
            correct = 1
        else:
            # predicted zero and was one
            confusion_matrix[0][1] += 1
    return correct, confusion_matrix

# Uses the distance between the correct simulated image and the predicted simulated image to determine correctness. Keeps track of said distances 
def distance_absolute(correct_pred_dist, num_predicted_correct, num_predicted_incorrect, distances, i):
    correct = 0
    if correct_pred_dist == 0:
        correct = 1
        num_predicted_correct.setdefault(real_label, 0)
        num_predicted_correct[real_label] += 1
    else:
        num_predicted_incorrect.setdefault(pred_label, 0)
        num_predicted_incorrect[pred_label] += 1

    distances[correct_pred_dist] = i+1

    return correct, num_predicted_correct, num_predicted_incorrect, distances

# Correct if the label rank is either perfect or just out of phase
def label_rank(label_rank, pred_label, correct_vector):
    correct = 0
    if label_rank[pred_label] == 0:
        # Wrong
        correct_vector[0] += 1
    elif label_rank[pred_label] == 1:
        # True correct
        correct_vector[1] += 1
        correct = 1
    elif label_rank[pred_label] == 2:
        # Imperfect correct
        correct_vector[2] += 1
        correct = 1
    elif label_rank[pred_label] == 3:
        # Wrong
        correct_vector[3] += 1
    return correct, correct_vector

# Turns confusion matrix into readable table
def confusion_matrix_to_table(confusion_matrix):
    table = [
        {"Real 0": confusion_matrix[0][0], "Real >0": confusion_matrix[1][0]},
        {"Real 0": confusion_matrix[0][1], "Real >0": confusion_matrix[1][1]}]
    
    pd_table = pd.DataFrame(table, index = ["Predicted 0", "Predicted >0"])
    return pd_table

# Returns the k largest distances and the k largest distances with their indices
def get_k_largest_dists(distances, k):
    distance_keys = list(distances)
    distance_keys.sort(reverse=True)
    distance_keys = distance_keys[0:k]
    k_largest_dists = []

    for key in distance_keys:
        round_key = round(key, 3)
        k_largest_dists.append((round_key, distances[key]))
    return distance_keys, k_largest_dists

# Prints out each label and their accuracy
def print_accuracy_by_label(num_predicted_correct, num_predicted_incorrect):
    incor_pred_keys = list(num_predicted_incorrect)
    incor_pred_keys.sort()

    print("\nAccuracy by label:")
    for key in incor_pred_keys:
        correct_label_count = num_predicted_correct[key] if key in num_predicted_correct else 0
        label_count = num_predicted_incorrect[key] + correct_label_count
        print(f"{key} accuracy: {correct_label_count/label_count}")

# Finds image in the dataset based on its index (quite inefficient, definitely should find a better way to do this)
def find_img(img_idx):
    for _i, image in enumerate(dataset):
        if _i == img_idx:
            return image
        
# Adapted from sample_to_image from src/npsv3/images/example
def sample_to_image(sample, with_simulations=False, margin=10):
    real_img = Image.fromarray(sample["image.npy.gz"][:, :, [0, 1, 5]])
    sim_imgs = sample["sim.images.npy.gz"]

    width, height = real_img.size
    # Creates a new image based on the number of simulated images plus some padding
    # The image does not need to be this large if simulated images are not printed, but I did not implement this functionality
    img = Image.new(real_img.mode, (width + (len(sim_imgs)-1) * (width + margin), 2 * height + margin))
    label = sample["label.cls"]
    img.paste(real_img, (label * width + label * margin, 0))
    
    if with_simulations:
        for _i, sim_img in enumerate(sim_imgs):
            sim_img = Image.fromarray(sim_img[0][:, :, [0, 1, 5]])
            img.paste(sim_img, (_i * width + _i * margin, height + margin))
    return img

# Takes in a list of [key], a dictionary of {key : index}, and a list of [label]
def display_images(sample_list, label_list):
    job_num = sys.argv[1]
    # Images have certain identifiable information, notably the key is also printed, which is usually distance between the correct and predicted image
    print("\nImages: Index | Image Label | Predicted Label | Region", flush=True)
    for img_idx, sample in sample_list:
        # print(sample["label.cls"])
        img_label = label_list[img_idx]
        # Does not highlight the predicted simulated image
        img = sample_to_image(sample, with_simulations=True)
        # Saves images to folder created by bash script
        img.save(f"ada_output/{job_num}/images/output_image_{img_idx}.png")
        print(f"{img_idx} | {sample["label.cls"]} | {img_label} | {sample["__key__"]}")

# Not an amazing dataset... Should probably try to find a better one
# images_path = "/storage/mlinderman/projects/sv/npsv3-experiments/training/hg002v1.1.dipcall.passing.sv.hg38.images/generator=coverage,pileup=unphased,simulation.replicates=1/images-{0000..0015}.tar"
images_path = "/storage/mlinderman/projects/sv/npsv3-experiments/training/hgsvc3-hprc-2024-02-23.dipcall.passing.hg38.eval-images/NA19240/generator=coverage,pileup=unphased_variable,simulation.replicates=1/images-{0000..0015}.tar"
dataset = wds.WebDataset(images_path, shardshuffle=False).decode()
model_name = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model = model.to(device)
model.eval()

print(f"Model: {model_name}")
print(f"Dataset: {images_path}")

# Maximum number of images that will be predicted
n = 1
# Targeted number of incorrect images displayed after evaluation
k = 0
# Defines how many variants will be classified before another is printed
printerval = 50
# Allows the model to start evaluation at a later index
start_index = 0
print(f"n = {n}")
# Keeps track of the number of correct predictions
total_correct = 0
# The index at which entries in k_predicted_incorrect will be replaced
repl_idx = random.randint(1, k)

# Creates matrix that will be added to confusion matrix. x is predicted and y is real, so (0, 1) means predicted 0 but was 1
confusion_matrix = np.zeros((2,2))
correct_vector = np.zeros(4)
distances = {}
num_predicted_correct = {}
num_predicted_incorrect = {}
k_predicted_incorrect = []
pred_labels = []

print("\nStarting prediction:", flush=True)
start_time = time.time()

for i, sample in enumerate(dataset):
    if i >= n:
        break

    if i < start_index:
        continue

    real_label = sample["label.cls"]
    pred_label, dist, correct_pred_dist, inputs = predict_sample_label(sample, model, "euclid_dist")
    pred_labels.append(pred_label)

    correct, correct_vector = label_rank(sample["label.rank.npy"], pred_label, correct_vector)
    if correct != 1:
        if len(k_predicted_incorrect) <= k:
            k_predicted_incorrect.append((i, sample))
        # If certain chance (which I would like to be k/num_incorrect) then replace an entry in the set of k incorrect predicted images
        elif random.randint(1, 1000) == 1:
            # replace a random entry in the list with a new one
            k_predicted_incorrect[repl_idx%k] = (i, sample)
            repl_idx += 1

    '''
    Keeping track of all predicted incorrect, each incorrect sample has a k/num_incorrect chance of being chosen.
    We could instead keep track of k predicted incorrect but replace them with a k/num_incorrect chance.
    The issue with this approach is it requires us to know the number of incorrect samples ahead of time.
    We could predict this number by multiplying the current accuracy by the number of samples, but the number of samples is not necessarily known either
    I don't think it is possible to know the number of samples in a tar file without counting them manually
    We know the number of samples we have gone through so far and the number of those that were incorrect
    Is there a fast way to count the number of samples in the dataset? Maybe a way to iterate through them without loading every sample into memory?
    '''

    # correct, confusion_matrix = presence_absence(real_label, pred_label, confusion_matrix)

    # correct, num_predicted_correct, num_predicted_incorrect, distances = distance_absolute(correct_pred_dist, num_predicted_correct, num_predicted_incorrect, distances, i)

    total_correct += 1 if correct == 1 else 0

    num_sim_imgs = len(sample["sim.images.npy.gz"])

    if (i+1) % printerval == 0:
        print(f"Image {i+1} | Image width: {inputs.pixel_values.shape[3]} | Real label: {real_label} Predicted label: {pred_label} Num sim images: {num_sim_imgs} Percent correct: {round((total_correct/((i-start_index)+1))*100, 2)}%", flush=True)
        # # print(inputs.pixel_values.shape)
        # real_img = Image.fromarray(sample["image.npy.gz"][:, :, [0, 1, 5]])
        # real_img.save(f"ada_output/{sys.argv[1]}/images/real_image_{i+1}.png")
        # torchvision.utils.save_image(inputs.pixel_values, f"ada_output/{sys.argv[1]}/images/processed_image_{i+1}.png")


end_time = time.time()

print(f"\nTime to run: {end_time-start_time} seconds", flush=True)
print(f"Accuracy: {(total_correct/(i-start_index))*100}%")

confusion_table = confusion_matrix_to_table(confusion_matrix)
# display(confusion_table)

print(f"zeros: {correct_vector[0]}")
print(f"ones: {correct_vector[1]}")
print(f"twos: {correct_vector[2]}")
print(f"threes: {correct_vector[3]}\n")

# k_largest_dists, k_worst_pred = get_k_largest_dists(distances, k)
# print("k largest distances:",k_worst_pred)

# k_random_dists = random.sample(list(distances), k)

# img_idx = 10
# sample = find_img(img_idx)
# image = sample_to_image(sample, with_simulations=True)
# job_num = sys.argv[1]
# image.save(f"ada_output/{job_num}/images/output_image_{img_idx}.png")

# print_accuracy_by_label(num_predicted_correct, num_predicted_incorrect)

display_images(k_predicted_incorrect, pred_labels)