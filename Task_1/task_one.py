import os
from collections import Counter
from PIL import Image
from PIL.ExifTags import TAGS

main_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(main_dir)

labels = os.listdir(os.path.join(parent_dir, "labels"))
samples = os.listdir(os.path.join(parent_dir, "samples"))

def compare_labels_and_samples():
    print(f"Total labels: {len(labels)}")
    print(f"Total samples: {len(samples)}")
    if len(labels) == len(samples):
        return "Similar quantity of samples and labels."
    else:
        short_label = [filename.replace('_ndvi_', '_') for filename in labels]
        short_sample = [filename.replace('_img_', '_') for filename in samples]
        if len(labels) > len(samples):
            print("Missing the following samples:")
            missing = set(short_sample).difference(set(short_sample))
            return missing
        if len(labels) < len(samples):
            print("Missing the following labels:")
            missing = set(short_sample).difference(set(short_label))
            return missing

def get_exif(filepath):
    properties = {
        'error': None,
        'mode': None,
        'width': None,
        'height': None,
        'bands': None,
        'resolution': None,
        'exif': {}
    }
    try:
        img = Image.open(filepath)
        properties['mode'] = img.mode
        properties['width'] = img.width
        properties['height'] = img.height
        properties['bands'] = len(img.getbands())
        properties['resolution'] = (img.width, img.height)
        exif_data = img.getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if isinstance(value, bytes):
                    try:
                        value = value.decode('utf-8', errors='ignore')
                    except:
                        value = str(value)
                properties['exif'][tag] = value
    except Exception as e:
        properties['error'] = str(e)

    return properties

def verify_metadata():
    sample_metadata = {}  
    label_metadata = {}

    sample_mode_counter = Counter()
    sample_bands_counter = Counter()
    sample_resolution_counter = Counter()
    
    label_mode_counter = Counter()
    label_bands_counter = Counter()
    label_resolution_counter = Counter()
    
    for filename in samples:
        filepath = os.path.join("samples", filename)
        properties = get_exif(filepath)
        sample_metadata[filename] = properties
        if not properties['error']:
            sample_mode_counter[properties['mode']] += 1
            sample_bands_counter[properties['bands']] += 1
            sample_resolution_counter[properties['resolution']] += 1
    
    for filename in labels:
        filepath = os.path.join("labels", filename)
        properties = get_exif(filepath)
        label_metadata[filename] = properties
        if not properties['error']:
            label_mode_counter[properties['mode']] += 1
            label_bands_counter[properties['bands']] += 1
            label_resolution_counter[properties['resolution']] += 1

    sample_mode = sample_mode_counter.most_common(1)[0][0] if sample_mode_counter else None
    sample_bands = sample_bands_counter.most_common(1)[0][0] if sample_bands_counter else None
    sample_res = sample_resolution_counter.most_common(1)[0][0] if sample_resolution_counter else None
    
    label_mode = label_mode_counter.most_common(1)[0][0] if label_mode_counter else None
    label_bands = label_bands_counter.most_common(1)[0][0] if label_bands_counter else None
    label_res = label_resolution_counter.most_common(1)[0][0] if label_resolution_counter else None
    
    non_standard_samples = []
    for filename, props in sample_metadata.items():
        if props['error']:
            non_standard_samples.append(f"{filename} (ERROR: {props['error']})")
        elif props['resolution'] != sample_res:
            non_standard_samples.append(f"{filename} (resolution: {props['resolution']}, expected: {sample_res})")
        elif props['mode'] != sample_mode:
            non_standard_samples.append(f"{filename} (mode: {props['mode']}, expected: {sample_mode})")
        elif props['bands'] != sample_bands:
            non_standard_samples.append(f"{filename} (bands: {props['bands']}, expected: {sample_bands})")

    non_standard_labels = []
    for filename, props in label_metadata.items():
        if props['error']:
            non_standard_labels.append(f"{filename} (ERROR: {props['error']})")
        elif props['resolution'] != label_res:
            non_standard_labels.append(f"{filename} (resolution: {props['resolution']}, expected: {label_res})")
        elif props['mode'] != label_mode:
            non_standard_labels.append(f"{filename} (mode: {props['mode']}, expected: {label_mode})")
        elif props['bands'] != label_bands:
            non_standard_labels.append(f"{filename} (bands: {props['bands']}, expected: {label_bands})")
    
    if non_standard_samples:
        print(f"({len(non_standard_samples)} non-standard samples):")
        for f in non_standard_samples:
            print(f" - {f}")
    else:
        print("All sample files follow the same standard.")

    if non_standard_labels:
        print(f"({len(non_standard_labels)} non-standard labels):")
        for f in non_standard_labels:
            print(f"- {f}")
    else:
        print("All label files follow the same standard.")

    return sample_metadata, label_metadata

print(compare_labels_and_samples())
print('\n')
verify_metadata()