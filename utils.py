
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import textwrap

def sanitize_filename(filename):
    return re.sub('[^0-9a-zA-Z]+', '_', filename)

def plot_images_and_text(predicted_image1, predicted_image2, groundtruth_image, generated_text, gt_text, save_dir, task_name, input_texts, input_images):
    task_path = os.path.join(save_dir, task_name)
    if not os.path.exists(task_path):
        os.makedirs(task_path)
    max_width = 50  # adjust this value based on your needs

    fig, ax = plt.subplots()
    ax.imshow(predicted_image1)
    generated_text = generated_text.replace("###", "").replace("[IMG0]", "")
    wrapped_generated_text = textwrap.fill(generated_text, max_width)
    ax.set_title(wrapped_generated_text, pad=20)
    ax.axis('off')
    plt.savefig(os.path.join(task_path, f"generated.jpg"), bbox_inches='tight')
    plt.close(fig)

    gt_text = gt_text.replace("$", "\$")
    wrapped_gt = textwrap.fill(gt_text, max_width)
    if predicted_image2 is not None:
        fig, ax = plt.subplots()
        ax.imshow(predicted_image2)
        ax.set_title(wrapped_gt, pad=20)
        ax.axis('off')
        plt.savefig(os.path.join(task_path, f"sd_baseline.jpg"), bbox_inches='tight')
        plt.close(fig)

    if groundtruth_image is not None:
        fig, ax = plt.subplots()
        groundtruth_image = groundtruth_image.float().cpu().numpy().squeeze()
        groundtruth_image = np.transpose(groundtruth_image, (1, 2, 0))
        groundtruth_image = np.uint8(groundtruth_image*255)
        ax.imshow(groundtruth_image)
        ax.set_title(wrapped_gt, pad=20)
        ax.axis('off')
        plt.savefig(os.path.join(task_path, f"gt.jpg"), bbox_inches='tight')
        plt.close(fig)

    if len(input_texts):
        max_width = 30
        length = len(input_texts)
        if length > 1:
            fig, ax = plt.subplots(1, length, figsize=(10*length, 10))
            for i in range(length):
                if i < len(input_images):
                    ax[i].imshow(input_images[i])
                    ax[i].set_title(textwrap.fill(input_texts[i], max_width), fontsize=28)
                    ax[i].axis('off')
                else:
                    ax[i].text(0.5, 0.5, textwrap.fill(input_texts[i], max_width), horizontalalignment='center', verticalalignment='center', fontsize=28)
                    ax[i].axis('off')
        else:
            fig, ax = plt.subplots()
            ax.imshow(input_images[0])
            ax.set_title(textwrap.fill(input_texts[0], max_width), fontsize=28)
            ax.axis('off')
        plt.savefig(os.path.join(task_path, f"input.jpg"), bbox_inches='tight')
        plt.close(fig)

    return None
