## Problems

## Code implementation (5 points)
Pass test cases by implementing the functions in the `code` directory.

Your grade for this section is defined by the autograder. If it says you got an 80/100,
you get 4 points here.

## Free response questions (5 points)

Answer the following free response questions in a separate document, 
saved as a .pdf and **uploaded to Canvas**.

### Polynomial regression (total of 2 points, broken into 2 parts)

First, we will run experiments on the `PolynomialRegression` code you wrote.

#### Generate data 
Generate data (via the function you implemented `generate_regression_data`). Generate data from polynomial functions of varying degrees, from 1 to 10. 

#### Perform regression (1 point) 
For each of your 10 generated datasets:
   1. run your implementation of `PolynomialRegression` 10 times: One time for each degree, from 1:10. 
   2. create a single plot that shows mean squared error (vertical) of your regression as a function of degree (horizontal).
   
Now, pick a single data file (I suggest the one generated from a polynomial of degree 4, but it's your choice) to illustrate bias vs variance and the dangers of overfitting.
   1. Show a scatter plot of the dataset. Indicate its degree in the title of the plot.
   2. Plot 3 functions learned with`PolynomialRegression` over the scatter plot: A functioned learned by a lower degree (e.g. if the data was generated with degree 4, then show a learned function with, say... degree 1); a function learned by the same degree (e.g. data generated with degree 4 and learned with a degree 4 polynomial) and the function that showed the lowest error learned with a higer degree polynomial than was used to generate the data.

#### Describe what's going on here (1 point) 
In the plots of mean squared error vs degree, what patterns do you notice? Specifically
in relation to the actual degree polynomial used to generate the dataset. Explain why
using the highest degree polynomial possible may or may not be a good idea. Put your
answer in terms of *bias* and *variance*:

https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff

### Perceptron (total of 2 points)

The following datasets are provided:

```
data/blobs.json
data/circles.json
data/crossing.json
data/parallel_lines.json
data/transform_me.json
```

Note that you can upload these files to the website http://ml-playground.com/. This
website will let you see how different learning algorithms work on each dataset. This
is not required for points but might be illuminating and useful for checking your work!

#### For each of these 5 datasets (.2 points per data set):

1. Show a scatter plot of the data with each point colored according to its class.
   1. See load_json_data for code example.
2. Draw the linear separator learned by your perceptron on top of the scatter plot.
   1. Not all of the datasets will be separable, so this plot may look odd.
3. Did your perceptron work on the dataset? Why or why not? 

#### For the dataset `data/transform_me.json` (1 point):
1. Make sure the relevant test case in the function `transform_data` in `code/perceptron.py` passes.
2. Show two scatter plots of the data before transformation and after transformation.
3. Why did the transformation you implemented work?
4. Read about the kernel method (sometimes called the kernel trick):
   1. https://en.wikipedia.org/wiki/Kernel_method
5. Explain how the transformation you implemented relates to the kernel trick.
