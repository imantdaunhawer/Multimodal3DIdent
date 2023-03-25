"""Render images for the Multimodal3DIdent dataset using Blender.

This script assumes the following values of latent variables and transforms
them for the input to Blender as follows.

| Variable                     | Values         | Blender Input                   | Transformation  |
| --------                     | ------         | -------------                   | --------------  |
| object shape                 | {0, 1, ..., 6} | {"Teapot", "Hare", ...}         | linear          |
| object x-position            | {0, 1, 2}      | {-3, 0, 3}                      | linear          |
| object y-position            | {0, 1, 2}      | {-3, 0, 3}                      | linear          |
| object z-position            | {0}            | {0}                             | none            |
| object alpha-rotation        | [0, 1]         | [-pi/2, pi/2]                   | linear          |
| object beta-rotation         | [0, 1]         | [-pi/2, pi/2]                   | linear          |
| object gamma-rotation        | [0, 1]         | [-pi/2, pi/2]                   | linear          |
| object color (hue value)     | [0, 1]         | (R, G, B)                       | HSV to RGB      |
| spotlight position           | [0, 1]         | (x, y) in [-4, 4]               | sine and cosine |
| spotlight color (hue value)  | [0, 1]         | (R, G, B)                       | HSV to RGB      |
| background color (hue value) | [0, 1]         | (R, G, B)                       | HSV to RGB      |

More precisely, colors and spotlight position are computed as follows:
- Hue values in HSV are transformed to RGB values. For object color we use
  hsv_to_rgb(x) with x in [0, 1], whereas for spotlight and background colors,
  we use hsv_to_rgb(x) with x in [-1/4, 1/4].

- Spotlight position (theta) is transformed to coordinates (x, y) on a semicircle, where
  - x = 4 * sin(theta'), theta' in [-pi/2, pi/2], and
  - y = 4 * cos(theta'), theta' in [-pi/2, pi/2].

The code builds on the following projects:
- https://github.com/brendel-group/cl-ica
- https://github.com/ysharma1126/ssl_identifiability
- https://github.com/facebookresearch/clevr-dataset-gen
"""

import argparse
import colorsys
import csv
import os
import pathlib
import site
import sys
import warnings

import numpy as np

# define constants
SHAPES = {
    0: 'Teapot',
    1: 'Hare',
    2: 'Dragon',
    3: 'Cow',
    4: 'Armadillo',
    5: 'Horse',
    6: 'Head'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--n-batches", type=int, default=None)
    parser.add_argument("--batch-index", type=int, default=None)
    parser.add_argument("--no-spotlights", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--material-name", default="Rubber", type=str)
    parser.add_argument("--shape-names", nargs="+", type=str)
    parser.add_argument("--save-scene", action="store_true")
    argv = render_utils.extract_args()
    args = parser.parse_args(argv)
    return parser, args


def main():

    # parse args
    _, args = parse_args()

    # define output folder
    args.output_folder = pathlib.Path(args.output_folder).absolute()

    # create output directory
    os.makedirs(os.path.join(args.output_folder, "images"), exist_ok=True)

    # load latents as dict
    csvpath = os.path.join(args.output_folder, "latents_image.csv")
    with open(csvpath, mode='r') as f:
        reader = csv.reader(f, delimiter=",")
        keys = [val for val in next(reader)]  # first row in csv is header
        latents_image = {k: [] for k in keys}
        for row in reader:
            for k, val in zip(keys, row):
                latents_image[k].append(float(val))
    num_samples = len(latents_image["object_shape"])

    # determine indices according to batch_index
    if args.batch_index:
        assert args.batch_index < args.n_batches
        indices = np.array_split(np.arange(num_samples), args.n_batches)[args.batch_index]
    else:
        indices = np.arange(0, num_samples)

    # rendering loop
    # --------------
    for i, ix in enumerate(indices):

        # initialize renderer
        object_shape = latents_image["object_shape"][ix]
        object_shape_ix = SHAPES[object_shape]
        initialize_renderer(
            object_shape_ix,
            args.material_name,
            not args.no_spotlights,
            render_tile_size=256 if args.use_gpu else 64,
            use_gpu=args.use_gpu)

        # create filename
        filename = str(ix).zfill(int(np.ceil(np.log10(num_samples)))) + ".png"
        filepath = os.path.join(args.output_folder, "images", filename)
        if os.path.exists(filepath):
            warnings.warn(f"Overwriting existing file '{filepath}'")

        # render sample
        current_latents = {k: v[ix] for k, v in latents_image.items()}
        render_sample(current_latents, args.material_name, not args.no_spotlights, filepath, args.save_scene)

    print(f"Done. Saved images to '{args.output_folder}/images/'.")


def initialize_renderer(
    shape_name,
    material_name,
    include_lights=True,
    width=224,
    height=224,
    render_tile_size=64,
    use_gpu=False,
    render_num_samples=512,
    render_min_bounces=8,
    render_max_bounces=8,
    ground_texture=None,
):
    """Initialize renderer and base scene"""

    base_path = pathlib.Path(__file__).parent.absolute()

    # Load the main blendfile
    base_scene = os.path.join(base_path, "assets", "scenes", "base_scene_equal_xyz.blend")
    bpy.ops.wm.open_mainfile(filepath=base_scene)

    # Load materials
    material_dir = os.path.join(base_path, "assets", "materials")
    render_utils.load_materials(material_dir)

    # Load segmentation node group
    # node_path = 'data/node_groups/NodeGroupMulti4.blend'
    segm_node_path = os.path.join(base_path, "assets/node_groups/NodeGroup.blend")
    with bpy.data.libraries.load(segm_node_path) as (data_from, data_to):
        data_to.objects = data_from.objects
        data_to.materials = data_from.materials
        data_to.node_groups = data_from.node_groups
    segm_node_mat = data_to.materials[0]
    segm_node_group_elems = (
        data_to.node_groups[0].nodes["ColorRamp"].color_ramp.elements
    )

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.resolution_x = width
    render_args.resolution_y = height
    render_args.resolution_percentage = 100
    render_args.tile_x = render_tile_size
    render_args.tile_y = render_tile_size
    if use_gpu == 1:
        # Blender changed the API for enabling CUDA at some point
        if bpy.app.version < (2, 78, 0):
            bpy.context.user_preferences.system.compute_device_type = "CUDA"
            bpy.context.user_preferences.system.compute_device = "CUDA_0"
        else:
            # Mark all scene devices as GPU for cycles
            bpy.context.scene.cycles.device = "GPU"

            for scene in bpy.data.scenes:
                scene.cycles.device = "GPU"
                scene.render.resolution_percentage = 100
                scene.cycles.samples = render_num_samples

            # Enable CUDA
            bpy.context.preferences.addons[
                "cycles"
            ].preferences.compute_device_type = "CUDA"

            # Enable and list all devices, or optionally disable CPU
            for devices in bpy.context.preferences.addons[
                "cycles"
            ].preferences.get_devices():
                for d in devices:
                    d.use = True
                    if d.type == "CPU":
                        d.use = False

    # Some CYCLES-specific stuff
    bpy.data.worlds["World"].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = render_max_bounces
    if use_gpu == 1:
        bpy.context.scene.cycles.device = "GPU"

    # activate denoising to make spot lights look nicer
    bpy.context.scene.view_layers["RenderLayer"].cycles.use_denoising = True
    bpy.context.view_layer.cycles.use_denoising = True

    # disable reflections
    bpy.context.scene.cycles.max_bounces = 0

    # Now add objects and spotlights
    add_objects_and_lights(shape_name, material_name, include_lights, base_path)

    max_object_height = max(
        [max(o.dimensions) for o in bpy.data.objects if "Object_" in o.name]
    )

    # Assign texture material to ground
    if ground_texture:
        render_utils.add_texture("Ground", ground_texture)
        # TODO: change z location if texture is used
    else:
        objs = bpy.data.objects
        objs.remove(objs["Ground"], do_unlink=True)

        bpy.ops.mesh.primitive_plane_add(size=1500, location=(0, 0, -max_object_height))
        bpy.context.object.name = "Ground"

        bpy.data.objects["Ground"].select_set(True)
        bpy.context.view_layer.objects.active = bpy.data.objects["Ground"]

        # bpy.data.objects["Ground"].data.materials.clear()
        render_utils.add_material("Rubber", Color=(0.5, 0.5, 0.5, 1.0))

    # Segmentation materials and colors
    n_objects = 1  # fixed to one
    segm_node_mat.node_tree.nodes["Group"].inputs[1].default_value = n_objects
    segm_mat = []
    segm_color = []
    for i in range(n_objects + 1):
        segm_node_mat.node_tree.nodes["Group"].inputs[0].default_value = i
        segm_mat.append(segm_node_mat.copy())
        segm_color.append(list(segm_node_group_elems[i].color))


def add_objects_and_lights(shape_name, material_name, add_lights, base_path):

    shapes_path = os.path.join(base_path, "assets", "shapes")
    print("Adding object", shape_name, material_name)
    # add object
    object_name = render_utils.add_object(
        shapes_path, f"Shape{shape_name}", "Object_0", 1.5, (0.0, 0.0, 0.0)
    )

    bpy.data.objects[object_name].data.materials.clear()
    render_utils.add_material(
        material_name, bpy.data.objects[object_name], Color=(0.0, 0.0, 0.0, 1.0)
    )

    if add_lights:
        # add spotlight focusing on the object
        # create light datablock, set attributes
        spotlight_data = bpy.data.lights.new(
            name="Spotlight_Object_0", type="SPOT"
        )
        spotlight_data.energy = 3000  # 10000, 10000 could be too bright
        spotlight_data.shadow_soft_size = 0.5
        spotlight_data.spot_size = 35 / 180 * np.pi
        spotlight_data.spot_blend = 0.1
        spotlight_data.falloff_type = "CONSTANT"

        spotlight_data.contact_shadow_distance = np.sqrt(3) * 3
        # create new object with our light datablock
        spotlight_object = bpy.data.objects.new(
            name="Spotlight_Object_0", object_data=spotlight_data
        )
        # link light object
        bpy.context.collection.objects.link(spotlight_object)

        spotlight_object.location = (7, 7, 7)

        ttc = spotlight_object.constraints.new(type="TRACK_TO")
        ttc.target = bpy.data.objects[object_name]
        ttc.track_axis = "TRACK_NEGATIVE_Z"
        # we don't care about the up_axis as long as it is different than TRACK_Z
        ttc.up_axis = "UP_X"

        # update scene, if needed
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()


def update_objects_and_lights(latents, material_name, update_lights):
    """Parse latents and update the object(s) position, rotation and color
    as well as the spotlight's position and color."""

    # find correct object name
    object_name = None
    for obj in bpy.data.objects:
        if obj.name.endswith(f"Object_{0}"):
            object_name = obj.name
            break
    assert object_name is not None

    # update object location and rotation
    object = bpy.data.objects[object_name]
    object.location = (
        latents["object_ypos"] * 3 - 3,  # visually, this is the vertical
        latents["object_xpos"] * 3 - 3,  # visually, this is the horizontal
        latents["object_zpos"])          # visually, this is the depth/distance
    print('obj loc', object.location)
    # scale each rotation from [0, 1] to [-pi/2, pi/2]
    rotation_euler = [latents[k] * np.pi - 0.5 * np.pi for k in
                      ("object_alpharot", "object_betarot", "object_gammarot")]
    object.rotation_euler = tuple(rotation_euler)

    # update object color
    rgb_object = colorsys.hsv_to_rgb(latents["object_color"], 1.0, 1.0)
    rgba_object = rgb_object + (1.0,)
    render_utils.change_material(
        bpy.data.objects[object_name].data.materials[-1], Color=rgba_object)

    if update_lights:
        # update light color
        hue_light = (latents["spotlight_color"] - 0.5) / 2.  # scale from [0, 1] to [-1/4, 1/4]
        rgb_light = colorsys.hsv_to_rgb(hue_light, 0.8, 1.0)
        bpy.data.objects["Spotlight_Object_0"].data.color = rgb_light
        # update light location
        spotlight_pos = latents["spotlight_pos"] * np.pi - 0.5 * np.pi  # scale to [-pi/2, pi/2]
        max_object_size = max([max(o.dimensions) for o in bpy.data.objects if "Object_" in o.name])
        bpy.data.objects["Spotlight_Object_0"].location = (
            4 * np.sin(spotlight_pos), 4 * np.cos(spotlight_pos), 6 + max_object_size, )


def render_sample(latents, material_name, include_lights, output_filename, save_scene):
    """Update the scene based on the latents and render the scene and save as an image."""

    # set output path
    bpy.context.scene.render.filepath = output_filename

    # set objects and lights
    update_objects_and_lights(latents, material_name, include_lights)
    hue_background = (latents["background_color"] - 0.5) / 2.  # scale from [0, 1] to [-1/4, 1/4]
    rgb_background = colorsys.hsv_to_rgb(hue_background, 0.6, 1.0)
    rgba_background = rgb_background + (1.0,)
    render_utils.change_material(
        bpy.data.objects["Ground"].data.materials[-1], Color=rgba_background)

    # set scene background
    bpy.ops.render.render(write_still=True)

    # just for debugging
    if save_scene:
        bpy.ops.wm.save_as_mainfile(
            filepath=f"scene_{os.path.basename(output_filename)}.blend")


if __name__ == "__main__":
    base_path = pathlib.Path(__file__).parent.absolute()

    INSIDE_BLENDER = True
    try:
        import bpy
        import bpy_extras
        from mathutils import Vector
    except ImportError as e:
        INSIDE_BLENDER = False
    if INSIDE_BLENDER:
        try:
            import render_utils
        except ImportError as e:
            try:
                print("Could not import render_utils.py; trying to hot patch it.")
                site.addsitedir(base_path)
                import render_utils
            except ImportError as e:
                print("\nERROR")
                sys.exit(1)

    if INSIDE_BLENDER:
        main()
    elif "--help" in sys.argv or "-h" in sys.argv:
        parser, _ = parse_args()
        parser.print_help()
    else:
        print("This script is intended to be called from blender like this:")
        print()
        print(
            "blender --background --python generate_3dident_dataset_images.py -- [args]"
        )
        print()
        print("You can also run as a standalone python script to view all")
        print("arguments like this:")
        print()
        print("python render_images.py --help")
