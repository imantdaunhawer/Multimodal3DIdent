"""Generate latents for the Multimodal3DIdent dataset."""

import argparse
import colorsys
import os
import random

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import CSS4_COLORS, TABLEAU_COLORS, XKCD_COLORS


class ColorPalette(object):
    """Color palette with utility functions."""

    def __init__(self, palette):
        """
        Initialize color palette.

        Args:
            palette (dict): dictionary of color-names to hex-values. For
                example: {"my_blue": "#0057b7", "my_yellow": "#ffd700"}.
        """
        # precompute rgb to name/hex for all keys and values in the palette
        self.rgb_to_name = {self.hex_to_rgb(v): k for k, v in palette.items()}
        self.rgb_to_hex = {self.hex_to_rgb(v): v for k, v in palette.items()}
        self.palette = palette

    def nearest_neighbor(self, rgb_value, return_name=True):
        """Given an rgb-value, find the nearest neighbor among the values of the palette."""
        assert len(rgb_value) == 3
        rgb_value_arr = np.array(rgb_value)
        min_dist = np.inf  # minimal distance
        rgb_nn = None      # rgb-value of the nearest neighbor
        for rgb_key in self.rgb_to_name.keys():
            dist = np.linalg.norm(np.array(rgb_key) - rgb_value_arr)  # euclidian distance
            if dist < min_dist:
                min_dist = dist
                rgb_nn = rgb_key
        if return_name:
            return self.rgb_to_name[rgb_nn]
        else:
            return rgb_nn

    @staticmethod
    def hex_to_rgb(hex_value):
        """Transform hex-code "#rrggbb" to rgb-tuple (r, g, b)."""
        rgb_value = matplotlib.colors.to_rgb(hex_value)
        return rgb_value

    @staticmethod
    def hue_to_rgb(hue_value):
        """Transform hue-value (between 0 and 1) to rgb-tuple (r, g, b)."""
        rgb_value = colorsys.hsv_to_rgb(hue_value, 1.0, 1.0)  # s and v are constant
        return rgb_value


def hue_to_colorname(object_hue, color_palettes):
    """Map hue value to a matching color name from a randomly sampled color palette.

    Args:
        object_hue (list or np.array): hue values in the interval [0, 1].
        color_palettes (list): list of color palettes (class `ColorPalette`).

    Returns:
        List of color names as an np.array of strings.
    """
    object_colorname = []
    for h in object_hue:
        j = np.random.randint(len(color_palettes))
        cp = color_palettes[j]
        rgb = cp.hue_to_rgb(h)
        colorname = cp.nearest_neighbor(rgb)  # color name of nearest neighbor
        object_colorname.append(colorname)
    return np.array(object_colorname)


def main():

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--n-points", type=int, required=True)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--position-dependent-color", action="store_true")
    args = parser.parse_args()

    # print args
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    # set all seeds
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # create output directory
    os.makedirs(args.output_folder, exist_ok=True)

    # define color palettes
    color_palettes = \
        [ColorPalette(x) for x in (TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS)]
    # map each color name to a unique integer
    color_names = []
    for cp in color_palettes:
        for k in cp.palette.keys():
            color_names.append(k)
    color_names = sorted(color_names)  # sort names to ensure the same order
    unique_names, color_indices = np.unique(color_names, return_inverse=True)
    colorname_to_index = \
        {name: index for (name, index) in zip(color_names, color_indices)}

    # define latent space
    # -------------------
    object_xpos = np.random.randint(0, 3, args.n_points)
    if args.position_dependent_color:
        # causal dependence of object-hue on object x-position
        object_hue = (object_xpos + np.random.rand(args.n_points)) / 3.
    else:
        object_hue = np.random.rand(args.n_points)
    object_colorname = hue_to_colorname(object_hue, color_palettes)
    object_colorindex = \
        np.array([colorname_to_index[cname] for cname in object_colorname])
    latents_image = {
        "object_shape": np.random.randint(0, 7, args.n_points),   # discrete, 7 values drawn uniformly
        "object_xpos": object_xpos,                               # discrete, 3 values drawn uniformly
        "object_ypos": np.random.randint(0, 3, args.n_points),    # discrete, 3 values drawn uniformly
        "object_zpos": np.zeros(args.n_points),                   # constant
        "object_alpharot": np.random.rand(args.n_points),         # continuous, uniform on [0, 1]
        "object_betarot": np.random.rand(args.n_points),          # continuous, uniform on [0, 1]
        "object_gammarot": np.random.rand(args.n_points),         # continuous, uniform on [0, 1]
        "object_color": object_hue,                               # continuous, uniform on [0, 1]
        "spotlight_pos": np.random.rand(args.n_points),           # continuous, uniform on [0, 1]
        "spotlight_color": np.random.rand(args.n_points),         # continuous, uniform on [0, 1]
        "background_color": np.random.rand(args.n_points),        # continuous, uniform on [0, 1]
    }
    latents_text = {
        "object_shape": latents_image["object_shape"],            # discrete, 7 values drawn uniformly
        "object_xpos": latents_image["object_xpos"],              # discrete, 3 values drawn uniformly
        "object_ypos": latents_image["object_ypos"],              # discrete, 3 values drawn uniformly
        "object_zpos": latents_image["object_zpos"],              # constant
        "object_color_name": object_colorname,                    # discrete, color name as string
        "object_color_index": object_colorindex,                  # discrete, color name as unique integer
        "text_phrasing": np.random.randint(0, 5, args.n_points),  # discrete, 5 values drawn uniformly
    }
    # NOTE The variables object_shape, object_xpos, and object_ypos are shared
    # between image and text latents. The variable object_color also has a
    # dependence between text and image latents, but the relation between
    # colors is not a 1-to-1 map, since the color palettes are sampled at
    # random from a set of multiple palettes (see function hue_to_colorname).

    # save latents to disk as csv files
    pd.DataFrame(latents_image).to_csv(os.path.join(args.output_folder, "latents_image.csv"), index=False)
    pd.DataFrame(latents_text).to_csv(os.path.join(args.output_folder, "latents_text.csv"), index=False)
    print(f"\nDone. Saved latents to '{args.output_folder}/'.")


if __name__ == "__main__":
    main()
