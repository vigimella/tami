import os, shutil, cv2, ast
from tqdm import tqdm
from PIL import Image, ImageStat
import numpy as np


def is_grayscale(img_path):
    im = Image.open(img_path).convert("RGB")
    stat = ImageStat.Stat(im)
    if sum(stat.sum) / 3 == stat.sum[0]:
        return True
    else:
        return False


def extract_coordinates(text):

    parts = text.strip("[]").split(",")
    return [int(parts[0]), int(parts[1])]


def high_on_image(over_img, smali_numbers, first_coordinates, second_coordinates, img_path, original_image):

    original_image = cv2.imread(original_image)

    or_height, or_width, or_channels = original_image.shape

    image = cv2.imread(over_img)

    ov_height, ov_width, ov_channels = image.shape

    scaling_factor_x = (ov_width / or_width)
    scaling_factor_y = (ov_height / or_height)

    img_name = over_img.split('/')[-1]
    circle_radius = 10

    for first_c, second_c, smali_number in zip(first_coordinates, second_coordinates, smali_numbers):

        top_left = ast.literal_eval(first_c)
        bottom_right = ast.literal_eval(second_c)

        center_x = int(((top_left[0] * scaling_factor_x) + (bottom_right[0] * scaling_factor_x)) // 2)
        center_y = int(((top_left[1] * scaling_factor_y) + (bottom_right[1] * scaling_factor_y)) // 2)

        text = str(smali_number)
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        # Calculate the text position based on the circle's position
        text_x = center_x - text_size[0] // 2
        text_y = center_y + text_size[1] // 2

        # Check if the circle is near the left or right border
        if center_x < text_size[0]:
            text_x = center_x + circle_radius
        elif center_x > (image.shape[1] - text_size[0]):
            text_x = center_x - text_size[0] - circle_radius

        # Check if the circle is near the top or bottom border
        if center_y < text_size[1]:
            text_y = center_y + circle_radius + text_size[1]
        elif center_y > (image.shape[0] - text_size[1]):
            text_y = center_y - circle_radius

        text_position = (text_x, text_y)

        cv2.circle(image, (center_x, center_y), circle_radius, (0, 0, 255), -1)
        cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    path_img_points = os.path.join(img_path, f'output_{img_name}')

    cv2.imwrite(path_img_points, image)

    return path_img_points


def generate_new_complete_map(heatmap_complete, overlay_img_points, img_path):

    new_complete_image_name = overlay_img_points.split('/')[-1].replace('output_', 'cati_complete_')
    new_image_path = os.path.join(img_path, new_complete_image_name)

    image1 = cv2.imread(heatmap_complete)

    image2 = cv2.imread(overlay_img_points)

    new_width = 400
    new_height = 400
    image2 = cv2.resize(image2, (new_width, new_height))

    combined_width = image1.shape[1] + new_width
    combined_height = max(image1.shape[0], new_height + 50)

    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    combined_image[:image1.shape[0], :image1.shape[1]] = image1
    combined_image[35:new_height + 35, image1.shape[1]:] = image2
    cv2.rectangle(combined_image, (image1.shape[1], 0), (combined_width, 25), (0, 0, 0), -1)
    cv2.putText(combined_image, 'CATI', (image1.shape[1] + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255))

    cv2.imwrite(new_image_path, combined_image)


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

        if item[0] != r and item[1] != g and item[2] != b:
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
        new_heatmap_path = change_heatmap_background(path_heatmap, path_output, r=93, g=166, b=10)
        transparency_heatmap_path = apply_transparency(path_heatmap=new_heatmap_path,
                                                       path_to_store_new_heatmap=path_output, r=93, g=166, b=10)

    else:
        transparency_heatmap_path = apply_transparency(path_heatmap=path_heatmap,
                                                       path_to_store_new_heatmap=path_output, r=0, g=0, b=0)

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


def get_smali_classes_highlighted(ov_image, path_store_final, path_legend_file, coordinates_list,
                                  complete_heatmap_path, original_image):
    smali_classes_identified = list()
    start_coordinates = list()
    end_coordinates = list()
    num_class = list()

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
                first_coordinates = str(range_start).replace('[', '(').replace(']', ')')
                second_coordinates = str(range_end).replace('[', '(').replace(']', ')')

                start_coordinates.append(first_coordinates)
                end_coordinates.append(second_coordinates)
                num_class.append(line_index)

                smali_classes_identified.append(f'ID: {line_index} - CLASS: {smali_class}')

    f.close()

    img_points = high_on_image(over_img=ov_image, smali_numbers=num_class, first_coordinates=start_coordinates,
                  second_coordinates=end_coordinates, img_path=path_store_final, original_image=original_image)

    generate_new_complete_map(heatmap_complete=complete_heatmap_path, overlay_img_points=img_points,
                              img_path=path_store_final)

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

        for highlight_heatmap in tqdm(os.listdir(highlight_heatmap_path)):

            current_image = highlight_heatmap.split('heatmap_')[-1].replace('-', '_').replace('.png', '')
            current_image_extension = current_image + f'.{file_extension}'
            current_image_path = os.path.join(dataset_image_path, current_image_extension)
            current_heatmap_path = os.path.join(highlight_heatmap_path, highlight_heatmap)
            current_class = dataset_image_path.replace(dataset_path + 'test/', '')

            image_obtained_from_apk_original = os.path.join(legend_file_path, f'{current_class}/{current_image}' + '.png')

            legend_file_path_img = os.path.join(legend_file_path, f'{current_class}/{current_image}' + '_legend.txt')

            if os.path.isfile(current_image_path) and not 'WRONG' in highlight_heatmap:

                print(f'Analyzing: {current_image}')

                temp_overlaid_img_dir = os.path.join(highlight_heatmap_path.replace('/highlights_heatmap', ''),
                                                     'temp_img_overlaid')

                if not os.path.isdir(temp_overlaid_img_dir):
                    os.mkdir(temp_overlaid_img_dir)

                final_smali_path = os.path.join(highlight_heatmap_path.replace('/highlights_heatmap', ''), 'smali')
                final_smali_path_file = os.path.join(final_smali_path, f'{current_image}.txt')
                img_overlaid = overlay_img(path_heatmap=current_heatmap_path, path_original=current_image_path,
                                           path_output=temp_overlaid_img_dir)
                if is_grayscale(current_image_path):
                    coordinates_list = get_coordinates(image_overlaid_path=img_overlaid, r_to_check=93, g_to_check=166,
                                                       b_to_check=10)
                else:
                    coordinates_list = get_coordinates(image_overlaid_path=img_overlaid, r_to_check=0, g_to_check=0,
                                                       b_to_check=0)

                current_complete_heatmap_path = highlight_heatmap_path.replace('highlights_heatmap',
                                                                               'complete/')
                current_complete_heatmap_path = current_complete_heatmap_path + f'complete_{current_image}.png'
                smali_classes_identified = get_smali_classes_highlighted(ov_image=img_overlaid,
                                                                         path_legend_file=legend_file_path_img,
                                                                         coordinates_list=coordinates_list,
                                                                         path_store_final=temp_overlaid_img_dir,
                                                                         complete_heatmap_path=current_complete_heatmap_path,
                                                                         original_image=image_obtained_from_apk_original)

                with open(final_smali_path_file, 'w') as final_report:
                    final_report.write(f'CATI tool final report for {current_image} \n')
                    final_report.write(f'The tool identified the following SMALI\'s class(es) \n')
                    for line in smali_classes_identified:
                        final_report.write(f'{line}\n')

                for file in os.listdir(temp_overlaid_img_dir):
                    if "cati" not in file:
                        file_path = os.path.join(temp_overlaid_img_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
