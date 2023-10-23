import os, shutil
from PIL import Image, ImageStat


def is_grayscale(img_path):
    im = Image.open(img_path).convert("RGB")
    stat = ImageStat.Stat(im)
    if sum(stat.sum) / 3 == stat.sum[0]:
        return True
    else:
        return False


def extract_coordinates(text):
    # get text from file_hash_legend.txt like "[x, y]"
    parts = text.strip("[]").split(",")
    return [int(parts[0]), int(parts[1])]


def change_heatmap_background(img_heatmap_path, path_to_store_new_heatmap, r, g, b):
    img_name = img_heatmap_path.split('/')[-1]
    img = Image.open(img_heatmap_path)
    img = img.convert("RGB")

    d = img.getdata()

    new_image = []
    for item in d:
        if item == (0, 0, 0):
            new_image.append((r, g, b))
        else:
            new_image.append(item)

    img.putdata(new_image)
    new_heatmap_path = os.path.join(path_to_store_new_heatmap, img_name)
    img.save(new_heatmap_path)

    return new_heatmap_path


def apply_transparency(path_heatmap, path_to_store_new_heatmap, r, g, b):
    list_color = list()
    newData = list()
    img = Image.open(path_heatmap)
    img = img.convert("RGBA")
    width, height = img.size

    colors = img.getcolors()

    for color in colors:
        list_color.append(color[1])

    list_color = list_color[:-1]
    datas = img.getdata()

    for item in datas:

        if not item[0] == r and item[1] == g and item[2] == b:
            newData.append((0, 0, 0, 0))
        else:
            newData.append(item)

    img_name = path_heatmap.split('/')[-1]
    new_heatmap_path = os.path.join(path_to_store_new_heatmap, img_name)
    img.putdata(newData)
    img.save(new_heatmap_path, "PNG")

    return new_heatmap_path


def overlay_img(path_original, path_heatmap, path_output):
    if is_grayscale(path_original):
        print('Your dataset is composed of grayscale images...')
        new_heatmap_path = change_heatmap_background(path_heatmap, path_output, r=255, g=0, b=0)
        transparency_heatmap_path = apply_transparency(path_heatmap=new_heatmap_path, path_to_store_new_heatmap=path_output, r=255, g=0, b=0)

    else:
        print('Your dataset is composed of RGB images...')
        transparency_heatmap_path = apply_transparency(path_heatmap=path_heatmap, path_to_store_new_heatmap=path_output, r=0, g=0, b=0)

    # get image's name

    img_name = path_original.split('/')[-1]

    # open images

    img_original = Image.open(path_original)
    img_original = img_original.convert('RGBA')
    img_heatmap = Image.open(transparency_heatmap_path)

    img_original.paste(img_heatmap, mask=img_heatmap)

    # save cropped image
    img_overlaid_save = os.path.join(path_output, 'overlaid_' + img_name)
    img_original.save(img_overlaid_save)

    print(f'Overlaid image saved successfully: {img_overlaid_save}')

    return img_overlaid_save


def get_coordinates(image_overlaid_path, r_to_check, g_to_check, b_to_check):
    coordinates_list = list()

    im = Image.open(image_overlaid_path)
    rgb_im = im.convert('RGB')
    width, height = im.size

    for x in range(width):
        for y in range(height):
            r, g, b = rgb_im.getpixel((x, y))
            if not (r == r_to_check and g == g_to_check and b == b_to_check):
                coordinates_list.append(f'[{x},{y}]')

    return coordinates_list


def get_smali_classes_highlighted(path_legend_file, coordinates_list):
    smali_classes_identified = list()

    with open(path_legend_file, 'r') as f:
        lines = f.readlines()

        for line_index, line in enumerate(lines):

            if line_index == 0:
                continue

            smali_found = False
            range_start = extract_coordinates(line.split(' ')[1])
            range_end = extract_coordinates(line.split(' ')[2].replace('\n', ''))
            smali_class = line.split(' ')[0]

            for point_text in coordinates_list:
                point = extract_coordinates(point_text)
                if (range_start[0] <= point[0] <= range_end[0]) and (range_start[1] <= point[1] <= range_end[1]):
                    smali_found = True
            if smali_found:
                smali_classes_identified.append(smali_class)

    f.close()

    return smali_classes_identified


def retrieve_smali_from_image(gradcam_path, dataset_path, legend_file_path):
    gradcam_class_names = list()
    dataset_class_names = list()

    for class_name in os.listdir(gradcam_path):
        gradcam_class_names.append(os.path.join(gradcam_path, f'{class_name}/highlights_heatmap'))
        dataset_class_names.append(os.path.join(dataset_path, f'test/{class_name}'))

    file_extension = ''

    for dataset_image_path in dataset_class_names:
        for file_dataset_image in os.listdir(dataset_image_path):
            extensions = file_dataset_image.split(".")[1:]
            file_extension = ".".join(extensions)
            file_extension = file_extension.lstrip(".")

    for highlight_heatmap_path, dataset_image_path in zip(gradcam_class_names, dataset_class_names):

        for highlight_heatmap in os.listdir(highlight_heatmap_path):
            current_image = highlight_heatmap.split('heatmap_')[-1].replace('-', '_').replace('.png', '')
            current_image_extension = current_image + f'.{file_extension}'
            current_image_path = os.path.join(dataset_image_path, current_image_extension)
            current_heatmap_path = os.path.join(highlight_heatmap_path, highlight_heatmap)
            current_class = dataset_image_path.replace(dataset_path + 'test/', '')

            print(f'CURRENT: {current_class}')

            legend_file_path_img = os.path.join(legend_file_path, f'{current_class}/{current_image}' + '_legend.txt')

            if os.path.isfile(current_image_path) and not 'WRONG' in highlight_heatmap:
                print(f'highlight_heatmap: {current_heatmap_path}')
                print(f'dataset image {current_image_path}')
                temp_overlaid_img_dir = os.path.join(highlight_heatmap_path.replace('/highlights_heatmap', ''), 'temp_img_overlaid')

                if not os.path.isdir(temp_overlaid_img_dir):
                    os.mkdir(temp_overlaid_img_dir)

                final_smali_path = os.path.join(highlight_heatmap_path.replace('/highlights_heatmap', ''), 'smali')
                final_smali_path_file = os.path.join(final_smali_path, f'{current_image}.txt')
                img_overlaid = overlay_img(path_heatmap=current_heatmap_path, path_original=current_image_path,
                                           path_output=temp_overlaid_img_dir)
                if is_grayscale(current_image_path):
                    coordinates_list = get_coordinates(image_overlaid_path=img_overlaid, r_to_check=255, g_to_check=0, b_to_check=0)
                else:
                    coordinates_list = get_coordinates(image_overlaid_path=img_overlaid, r_to_check=0, g_to_check=0, b_to_check=0)

                smali_classes_identified = get_smali_classes_highlighted(path_legend_file=legend_file_path_img,
                                                                         coordinates_list=coordinates_list)

                with open(final_smali_path_file, 'w') as final_report:
                    final_report.write(f'CATI tool final report for {current_image} \n')
                    final_report.write(f'The tool identified the following SMALI\'s class(es) \n')
                    for line in smali_classes_identified:
                        final_report.write(f'{line}\n')

                # shutil.rmtree(temp_overlaid_img_dir)
