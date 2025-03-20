import matplotlib
import platform
import matplotlib.pyplot as plt
import random

def set_japanese_font():
    if platform.system() == 'Windows':
        matplotlib.rcParams['font.family'] = 'MS Gothic'
    elif platform.system() == 'Darwin':  # Mac
        matplotlib.rcParams['font.family'] = 'Hiragino Sans'
    else:  # Linux
        matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
    
    matplotlib.rcParams['axes.unicode_minus'] = False



def display_class_samples(mapped, kana_chars, mapping):

    valid_class_indices = sorted(mapped.keys())
    mapping = {old: new for new, old in enumerate(valid_class_indices)}

    classes = sorted(mapped.keys())
    num_classes = len(classes)
    cols = 5
    rows = (num_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()
    
    for i, cls in enumerate(classes):
        if mapped[cls]:
            sample_img = random.choice(mapped[cls])
            new_index = mapping[cls]
            if new_index < len(kana_chars):
                title = f"Class {cls} ({kana_chars[new_index]})"
            else:
                title = f"Class {cls}"
            axes[i].imshow(sample_img.squeeze(), cmap='gray')
            axes[i].set_title(title, fontsize=10)
            axes[i].axis('off')
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()