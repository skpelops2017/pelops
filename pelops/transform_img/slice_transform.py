import os
import sys
import math
import shutil
import PIL.Image


def transform_1(input_image):
    """Original image size, 0 bg, Anchor (0, 0)"""
    in_img = input_image.copy()
    out_img = PIL.Image.new('RGB', (224, 224), color=0)
    out_img.paste(in_img, (0, 0))
    return out_img


def transform_2(input_image):
    """Original image size, 0 bg, Anchor centered"""
    in_img = input_image.copy()
    out_img = PIL.Image.new('RGB', (224, 224), color=0)
    anchor_x = 112 - int(in_img.size[0] / 2)
    anchor_y = 112 - int(in_img.size[1] / 2)
    out_img.paste(in_img, (anchor_x, anchor_y))
    return out_img


def transform_3(input_image):
    """Resampled image size on max integer multiple, 0 bg, Anchor centered, NEAREST interpolation"""
    in_img = input_image.copy()
    in_w, in_h = in_img.size
    mult_w = 224.0 / float(in_w)
    mult_h = 224.0 / float(in_h)
    factor = math.floor(min(mult_w, mult_h))
    in_img = in_img.resize((in_w * factor, in_h * factor), PIL.Image.NEAREST)
    out_img = PIL.Image.new('RGB', (224, 224), color=0)
    anchor_x = 112 - int(in_img.size[0] / 2)
    anchor_y = 112 - int(in_img.size[1] / 2)
    out_img.paste(in_img, (anchor_x, anchor_y))
    return out_img


def transform_4(input_image):
    """Resampled image size on max integer multiple, 0 bg, Anchor centered, BILINEAR interpolation"""
    in_img = input_image.copy()
    in_w, in_h = in_img.size
    mult_w = 224.0 / float(in_w)
    mult_h = 224.0 / float(in_h)
    factor = math.floor(min(mult_w, mult_h))
    in_img = in_img.resize((in_w * factor, in_h * factor), PIL.Image.BILINEAR)
    out_img = PIL.Image.new('RGB', (224, 224), color=0)
    anchor_x = 112 - int(in_img.size[0] / 2)
    anchor_y = 112 - int(in_img.size[1] / 2)
    out_img.paste(in_img, (anchor_x, anchor_y))
    return out_img


def transform_5(input_image):
    """Resampled image size on max integer multiple, 0 bg, Anchor centered, BICUBLIC interpolation"""
    in_img = input_image.copy()
    in_w, in_h = in_img.size
    mult_w = 224.0 / float(in_w)
    mult_h = 224.0 / float(in_h)
    factor = math.floor(min(mult_w, mult_h))
    in_img = in_img.resize((in_w * factor, in_h * factor), PIL.Image.BICUBIC)
    out_img = PIL.Image.new('RGB', (224, 224), color=0)
    anchor_x = 112 - int(in_img.size[0] / 2)
    anchor_y = 112 - int(in_img.size[1] / 2)
    out_img.paste(in_img, (anchor_x, anchor_y))
    return out_img


def transform_6(input_image):
    """Resampled image size on max integer multiple, 0 bg, Anchor centered, LANCZOS interpolation"""
    in_img = input_image.copy()
    in_w, in_h = in_img.size
    mult_w = 224.0 / float(in_w)
    mult_h = 224.0 / float(in_h)
    factor = math.floor(min(mult_w, mult_h))
    in_img = in_img.resize((in_w * factor, in_h * factor), PIL.Image.LANCZOS)
    out_img = PIL.Image.new('RGB', (224, 224), color=0)
    anchor_x = 112 - int(in_img.size[0] / 2)
    anchor_y = 112 - int(in_img.size[1] / 2)
    out_img.paste(in_img, (anchor_x, anchor_y))
    return out_img


def transform_7(input_image):
    """Resampled image size on max integer multiple, 0 bg, Anchor centered, anti aliased"""
    in_img = input_image.copy()
    in_w, in_h = in_img.size
    mult_w = 224.0 / float(in_w)
    mult_h = 224.0 / float(in_h)
    factor = math.floor(min(mult_w, mult_h))
    in_img = in_img.resize((in_w * factor, in_h * factor), PIL.Image.ANTIALIAS)
    out_img = PIL.Image.new('RGB', (224, 224), color=0)
    anchor_x = 112 - int(in_img.size[0] / 2)
    anchor_y = 112 - int(in_img.size[1] / 2)
    out_img.paste(in_img, (anchor_x, anchor_y))
    return out_img


if __name__ == '__main__':
    transforms = {var_name: fnc_ptr for var_name, fnc_ptr in locals().items() if var_name.startswith("transform_")}
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    if data_path is None or not os.path.isdir(data_path) or output_path is None:
        sys.exit(0)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    if data_path.endswith('/'):
        data_path = data_path[:-1]
    if output_path.endswith('/'):
        output_path = output_path[:-1]

    data_files = []
    for dirname, sub_names, file_names in os.walk(data_path):
        if dirname.endswith('/images'):
            data_files.extend([os.path.join(dirname, file_name) for file_name in file_names if file_name.endswith('.png')])
        if 'truth.txt' in file_names:
            data_files.append(os.path.join(dirname, 'truth.txt'))
    print("{} files to copy".format(len(data_files)))

    for name, transform_function in transforms.items():
        trans_dir = os.path.join(output_path, name)
        print(trans_dir)
        if not os.path.isdir(trans_dir):
            os.makedirs(trans_dir)

        with open(os.path.join(trans_dir, 'README'), 'w') as rhdl:
            rhdl.write(transform_function.__doc__)

        for item in data_files:
            out_item = item.replace(data_path, trans_dir)
            out_sub = os.path.dirname(out_item)
            if not os.path.isdir(out_sub):
                os.makedirs(out_sub)
            elif os.path.isfile(out_item):
                os.remove(out_item)

            if item.endswith('.png'):
                output_image = transform_function(PIL.Image.open(item))
                output_image.save(out_item)
            else:
                shutil.copy(item, out_item)
