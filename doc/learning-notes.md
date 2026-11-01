# Learning notes — what I learned building a CIFAR-10 classifier with an AI assistant

Written by me (Ibi), from re-reading the chat history in `doc/ai-chat-history/`.
Last updated 2026-11-02.

## Where I started and what I wanted this time

I already have some experience with neural networks and computer vision from uni. In those
assignments the main goal was the result, because I was going for marks, and the model was
mostly chosen for us. I usually spent most of my time on the model code and not much on
everything around it.

This time I wanted something different:

* focus on understanding how the model works and on tuning the parameters myself
* look at the accuracy and the figures and judge them myself, instead of asking Claude "is this
  good enough?" — but still leave room for Claude to suggest alternatives
* have a reason for the model I chose: I picked the small CNN because I wanted to keep it simple
  and leave the option open to add layers later for experiments

## How the files connect

I understand the pipeline much better now, and how each file has one job:

* `dataset.py` loads and prepares the data before it is fed to the model
* `model.py` is where the CNN architecture is built
* `train.py` trains the model and updates the weights
* `evaluate.py` checks the result and shows where the model is good or bad
* `predict.py` runs the trained model on my own photos

Things I learned from the data side:

* why we need to check the data split, that train and validation indices do not overlap, and
  that image sizes and labels are what we expect — before building any model
* the HTML layer visualisation helped me understand what happens inside each CNN layer
* I like the `layer_table` function because it checks the output dimension of each layer
  automatically instead of me calculating it by hand

## Mistakes Claude made, and what I took from them

Claude actually made some mistakes while building the code:

* it used the same dictionary key twice, which broke the data loader
* the in-place ReLU changed the values that were recorded for the layer visualisation, so the
  conv layers looked wrong on the page

Claude found both problems from its own testing and fixed them. I did not find these two bugs
myself, but I learned why testing is important and why a verification step before the next
piece of code is worth the time.

## A problem that was not in the code

I had a problem where importing PyTorch took almost one minute. After investigating, the cause was
that my virtual environment was on the Windows side of the filesystem (`/mnt/c/...`). I moved the
virtual environment to the Linux side of WSL and the import dropped to about one second.

So sometimes the problem is not the Python code, it is the environment itself.

## Checking the result myself

My baseline model got **77.70% validation accuracy**. I did not want to just trust the number, so I
checked the confusion matrix: 3,885 correct out of 5,000 images, which matches 77.70%.

I also tried my own photos instead of only CIFAR-10 images:

* my cat photo was predicted as **automobile with 96.3% confidence**
* after I cropped closer to the cat, the prediction changed to **cat with 54%**
* I tried a chimera image. I thought it would predict horse, cat or bird, but it predicted frog
  or airplane depending on the image version (transparent background vs white background)

This surprised me: **high confidence does not mean the prediction is correct.**

## What I found in the experiments

* I wanted to know whether learning rate and epochs have a linear relationship. From the
  experiment I understand it is not that simple, because the learning rate also changes during
  training (the scheduler).
* I compared the cosine, step and no-scheduler options. The scheduler was something I was not
  really familiar with before.

(Numbers and observations per run are in `doc/experiments.md`.)

## Where I can learn and experiment more

* I should review the evaluation code more carefully. I learned these metrics at uni and in
  assignments, but I do not truly understand yet how they work inside.
* If I had more time I would make a small example dataset where I already know the correct
  result, so I can compare the model output with the expected one.
* An idea for a future experiment: tell Claude directly to build everything at once, then compare
  it with this project where I kept the decisions. Would it cause more bugs, a different design,
  and how much unfamiliar code would it create?

## What I learned about working with AI

* Using AI is not only about getting code that works.
* Even if the AI explains something confidently, I still need to check.
* I do not need to find a mistake in every piece of code, but I should know why it works.
* Sometimes my expectation is wrong, and testing it is how I learn.

I think this project made me more careful with AI-generated code and with verifying the result.
