from wordcloud import (WordCloud, get_single_color_func)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)


class GroupedColorFunc(object):
    """Create a color function object which assigns DIFFERENT SHADES of
       specified colors to certain words based on the color to words mapping.

       Uses wordcloud.get_single_color_func

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)

def transform_format(val):
    if val == 0:
        return 255
    else:
        return val

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center >= radius
    return mask


def generate_wordcloud(non_zero_features, feature_weights, image_index, hidden_node_index, color, weight_threshold):
    text = ""
    color_to_words = {
        # words below will be colored with a green single color function
        '#00ff00': [],
        # will be colored with a red single color function
        'red': []
    }
    for idx in range(len(non_zero_features)):
        if np.abs(feature_weights[idx]) >= weight_threshold:
            word_frequency = int(abs(feature_weights[idx]) * 10000)
            for i in range(word_frequency):
                text += str(non_zero_features[idx]).replace(' ','_').replace('-','_') + " "
            if feature_weights[idx] >= weight_threshold:
                color_to_words['#00ff00'].append(str(non_zero_features[idx]).replace(' ','_').replace('-','_'))
            elif feature_weights[idx] <= -weight_threshold:
                color_to_words['red'].append(str(non_zero_features[idx]).replace(' ','_').replace('-','_'))
    if text != '':
        w = 500
        h = 500
        center = (int(w / 2), int(h / 2))
        radius = (h / 2) - 4
        circle_mask = create_circular_mask(h, w, center=center, radius=radius)*255
        # plt.imshow(circle_mask)
        # plt.show()

        # circle_mask = np.array(Image.open("img/circle.png"))[:,:,]

        # Transform your mask into a new one that will work with the function:
        # transformed_circle_mask = np.ndarray((circle_mask.shape[0], circle_mask.shape[1]), np.int32)
        #
        # for i in range(len(circle_mask)):
        #     transformed_circle_mask[i] = list(map(transform_format, circle_mask[i]))

        wc = WordCloud(width=w, height=h, collocations=False, background_color="white", mask=circle_mask, contour_width=8, contour_color=color, prefer_horizontal=1).generate(text)
        # Words that are not in any of the color_to_words values
        # will be colored with a grey single color function
        default_color = 'white'

        # Create a color function with single tone
        grouped_color_func = SimpleGroupedColorFunc(color_to_words, default_color)

        # Create a color function with multiple tones
        # grouped_color_func = GroupedColorFunc(color_to_words, default_color)

        # Apply our color function
        wc.recolor(color_func=grouped_color_func)

        # Plot
        plt.figure()
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad=0)
        # plt.show()
        plt.savefig(f"hidden_wordclouds/wc_{image_index}_{hidden_node_index}.png", bbox_inches='tight')
