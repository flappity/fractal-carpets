# fractal-carpets
Python fractal generation

This video (https://youtu.be/dPozHBz6Fqw) explains the basic idea behind this thing.

Essentially it starts with a 3x3 grid of 0's and a list of numbers their assigned rules.  

A rule for 0 might look like:
[[1, 0, 1],
 [1, 1, 1], 
 [0, 1, 1]]
 
The array is then expanded to 9x9. Looking at the original grid, each cell now corresponds to a 3x3 block in the new array.
 
So, if the first cell in the 3x3 grid is a 0, the first 3x3 block in the bigger grid will be replaced by the 3x3 rule assigned to it.
This is repeated for num_iter iterations - this can result in HUGE arrays very quickly, as the side lengths are progressive exponents of 3.
After six iterations, you're already at 2187x2187 cells.
 
This script does all of this with some customization - you can choose an arbitrary number of rules, how many iterations you want, how many different images to generate, etc.
 
The bottom line of the script is where you can play with these settings, or you can screw with the rest to make it do other magic.
There is crappy documentation within the script itself. I recommend sticking to 4-6 iterations - any more and it becomes way too big.
