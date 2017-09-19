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
            elif option == 'invert':
                if num_rules == 2:
                    ruleset[0] = np.random.randint(num_rules, size=(3, 3))
                    ruleset[1] = np.logical_not(ruleset[0]).astype(int)


        self.rules = ruleset

    def reset(self):
        self.array = np.zeros(shape=(self.x, self.y), dtype=np.int)

    def run(self, run_count, iter_count, randomize=1, option='None', num_rules=2, bw=0):
        """Runs the fractal generation.

        Args:
            run_count(int): Output this many images.

            iter_count(int): How many times to run the function, per image.
                5-6 is a safe number, too large and the output is huge

            randomize(int): 1: Generate new rules per image; 0: use hardcoded rules above

            option(str): Lets you force certain rules - see above - sort of useless with
                more than 3 colors.
                    "zero" forces rule_1 to be all 0's
                    "ones" forces rule_1 to be all 1's
                    "invert" only works with two colors - rule_1 is rule_0 inverted.

            num_rules(int): Choose how many rules/colors you want to use. Defaults to 2

            bw(int): 0 for a full (randomly generated) color selection, 1 for greyscale.
        """

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

#################################################################
# The line below is where you can play with the parameters of the fractal generation. See documentation above for
# further explanation.
# args: run_count, iter_count, keywords
# keywords: randomize(int), num_rules(int), bw(int), options(str) 'ones' 'zero' 'invert'
#
# WARNING: Don't make iter_count too high - images generated with 6 iterations are already huge enough.

fractal.run(1, 5, randomize=1, num_rules=10, bw=1)
