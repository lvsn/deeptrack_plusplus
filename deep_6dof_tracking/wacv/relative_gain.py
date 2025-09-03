import numpy as np


def str2digits(string):
    digits = [int(x) for x in string.split(" & ")]
    return digits


def relative_gain(scoreA, scoreB):
    return ((scoreA - scoreB)/ scoreA) * 100

if __name__ == '__main__':
    result_rgb = "82 & 175 & 242 & 289 & 325 & 376"
    result_moddrop = "74 & 174 & 224 & 252 & 276 & 321"

    result_rgb_digits = str2digits(result_rgb)
    result_moddrop_digits = str2digits(result_moddrop)
    total_rgb = sum(result_rgb_digits)
    total_moddrop = sum(result_moddrop_digits)

    print("Total rgb: {}".format(total_rgb))
    print("Total moddrop: {}".format(total_moddrop))
    print("Relative total : {}".format(relative_gain(total_rgb, total_moddrop)))

    result_rgb_digits = np.array(result_rgb_digits)
    result_moddrop_digits = np.array(result_moddrop_digits)

    relative_str = " & ".join([str("{0:.1f}".format(x)) for x in relative_gain(result_rgb_digits, result_moddrop_digits)])
    print(relative_str)