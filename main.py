import numpy as np
from matplotlib import pyplot as plt


class FractalArray:

    def __init__(self, x_initial, y_initial, ):
        self.x = x_initial
        self.y = y_initial
        self.array = np.zeros(shape=(self.x, self.y))
        self.rules = {0: np.array(
                          [[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]]),
                      1: np.array(
                          [[1, 1, 1],
                           [2, 0, 1],
                           [1, 1, 1]]),
                      2: np.array(
                          [[0, 1, 0],
                           [0, 0, 0],
                           [0, 1, 0]])}

        self.colors = {}

    def process_array(self):
        array_expanded = self.array.repeat(self.x, axis=0).repeat(self.y, axis=1)
        for x in range(self.array.shape[0]):
            for y in range(self.array.shape[1]):
                rule = self.rules[self.array[x, y]]
                sub_x = 3 * (x + 1) - 3
                sub_y = 3 * (y + 1) - 3
                rule_x = rule.shape[0]
                rule_y = rule.shape[1]
                array_expanded[sub_x:sub_x + rule_x, sub_y:sub_y + rule_y] = rule
        self.array = array_expanded

    def do_iterations(self, iter_count):
        iteration = 0
        while iteration < iter_count:
            self.process_array()
            iteration += 1

    def rand_rules(self, num_rules, option=None, bw=0):
        ruleset = {}

        for i in range(num_rules):
            ruleset[i] = np.random.randint(num_rules, size=(3, 3))
            if bw == 1:

                frac_bw = 1 / (num_rules - 1)
                black_amount = i * frac_bw
                # (0 1 2)
                # 0 50 100
                self.colors[i] = [black_amount, black_amount, black_amount]
            else:
                self.colors[i] = [np.random.random() / 2, np.random.random() / 2, np.random.random() / 2]
        if option:
            if option == 'zero':
                ruleset[1].fill('0')
            elif option == 'ones':
                ruleset[1].fill('1')
        self.rules = ruleset

    def reset(self):
        self.array = np.zeros(shape=(self.x, self.y), dtype=np.int)

    def run(self, run_count, iter_count, randomize=1, option='None', num_rules=2, bw=0):
        for a in range(run_count):
            self.reset()
            if bw == 1:
                ctype = 'greyscale'
            else:
                ctype = 'rgb'
            if randomize == 1:
                self.rand_rules(num_rules, option, bw)
                img_filename = "f_{0}r-{1}i_{2}.png"
                txt_filename = "f_{0}r-{1}i_{2}.txt"
            else:
                img_filename = "fractal_{1}{3}.png"
                txt_filename = "fractal_{1}{3}.txt"
            self.do_iterations(iter_count)

            f_txt = txt_filename.format(num_rules, iter_count, ctype, a)
            f_img = img_filename.format(num_rules, iter_count, ctype, a)

            rgb_array = np.zeros(shape=(self.array.shape[0], self.array.shape[1], 3))
            for i in range(num_rules):
                rgb_array[self.array == i] = self.colors[i]
            plt.imsave(f_img, rgb_array)

            with open(f_txt, 'w') as f:
                for key, value in self.rules.items():
                    f.write('%s:\n%s\n' % (key, value))


# Create the first stage of the fractal with x/y dimensions
fractal = FractalArray(3, 3)

# num_images, num_iterations, randomize, num_rules, option, bw
# num_images:       Number of different images to output
# num_iterations:   How many times to iterate - the image size increases very quickly, be careful
# randomize:        Will generate rules if set to 1; otherwise, uses rules in fractal.rules
# num_rules:        Number of "rules" or "colors" - each color has a rule for expansion
# option:           Hardcoded options that change rule 1 - relics of an old age
#                       zeros = rule_1 is ([0, 0, 0], [0, 0, 0], [0, 0, 0])
#                       ones = rule_1 is ([1, 1, 1], [1, 1, 1], [1, 1, 1])
# bw:               Generates b/w images - 2 rules = black/white, more colors = more shades of grey

fractal.run(1, 5, randomize=1, num_rules=10, bw=1)
