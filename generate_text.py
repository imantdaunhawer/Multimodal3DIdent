"""Generate text for the Multimodal3DIdent dataset."""

import argparse
import csv
import os

# define constants
XPOS = {
    0: 'left',
    1: 'center',
    2: 'right'}

YPOS = {
    0: 'top',
    1: 'mid',
    2: 'bottom'}

SHAPES = {
    0: 'teapot',
    1: 'hare',
    2: 'dragon',
    3: 'cow',
    4: 'armadillo',
    5: 'horse',
    6: 'head'}

PHRASES = {
    0: 'A {SHAPE} of "{COLOR}" color is visible, positioned at the {YPOS}-{XPOS} of the image.',
    1: 'A "{COLOR}" {SHAPE} is at the {YPOS}-{XPOS} of the image.',
    2: 'The {YPOS}-{XPOS} of the image shows a "{COLOR}" colored {SHAPE}.',
    3: 'At the {YPOS}-{XPOS} of the picture is a {SHAPE} in "{COLOR}" color.',
    4: 'At the {YPOS}-{XPOS} of the image, there is a "{COLOR}" object in the form of a {SHAPE}.'}


def main():

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-folder", type=str, required=True)
    args = parser.parse_args()

    # check if directory exists
    assert os.path.exists(args.output_folder)

    # create output directory
    os.makedirs(os.path.join(args.output_folder, "text"), exist_ok=True)
    output_path = os.path.join(args.output_folder, "text", "text_raw.txt")

    # load latents as dict
    csvpath = os.path.join(args.output_folder, "latents_text.csv")
    with open(csvpath, mode='r') as f:
        reader = csv.reader(f, delimiter=",")
        keys = [val for val in next(reader)]  # first row in csv is header
        latents_text = {k: [] for k in keys}
        for row in reader:
            for k, val in zip(keys, row):
                try:
                    latents_text[k].append(float(val))
                except ValueError:  # e.g., when val is a string
                    latents_text[k].append(val)
    num_samples = len(latents_text["object_shape"])

    # generate text
    with open(output_path, 'w') as f:
        for i in range(num_samples):
            j = int(latents_text["text_phrasing"][i])
            phrase = PHRASES[j]
            phrase = phrase.format(
                SHAPE=SHAPES[latents_text["object_shape"][i]],
                YPOS=YPOS[latents_text["object_ypos"][i]],
                XPOS=XPOS[latents_text["object_xpos"][i]],
                COLOR=latents_text["object_color_name"][i])
            if i < num_samples - 1:
                phrase = phrase + "\n"  # newline for all lines except the last
            f.write(phrase)
    print(f"Done. Saved text to '{output_path}'.")


if __name__ == "__main__":
    main()
