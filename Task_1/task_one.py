from pathlib import Path
import os
import PIL

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

label = os.listdir(os.path.join(parent_dir, "labels"))
sample = os.listdir(os.path.join(parent_dir, "samples"))

def compare_labels_and_samples():
    print(f"Total labels: {len(label)}")
    print(f"Total samples: {len(sample)}")
    if len(label) == len(sample):
        return "Similar Quantity of Samples and Labels."
    else:
        short_label = [filename.replace('_ndvi_', '_') for filename in label]
        short_sample = [filename.replace('_img_', '_') for filename in sample]
        if len(label) > len(sample):
            print("Missing the following samples:")
            missing = set(short_label).difference(set(short_sample))
            return missing
        if len(label) < len(sample):
            print("Missing the following labels:")
            missing = set(short_sample).difference(set(short_label))
            return missing

def analyze_metadata():
    # Not entirely sure how to do this right now... I'm thinking on how I can make sure that everything is the same size but I can't rely on the first one since maybe the first one is incongruent so i need to see.
    return 0

print(compare_labels_and_samples())