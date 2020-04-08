# Shoes-generator

## Create virtual environment  

Install tensorflow and keras in a virtual environment using python 3.6. Use `tensorflow-gpu` if you plan on running it on a GPU:  
`conda create --name Deep python=3.6 numpy tensorflow keras`

## Download dataset  
<http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images-square.zip>  
Courtesy of A. Yu and K. Grauman and Mark Stephenson  

Extract the zip at the root of this repository.
Navigate to it : 
```
cd ut-zap50k-images-square
```
The images are in many subfolders, extract them all into a folder called 'all_images' that you
create before:
```
mkdir all_images  
find . -type f -print0 | xargs -0 mv -t all_images 
```

## Project description
Our main idea for this project is to use deep learning techniques to generate new unseen images from a model that has been trained on an image dataset.
For the scale of this project however, it is wise to restrict ourselves to one type of object rather than a multitude of different ones. Our preference lies on a shoe dataset, as the process of designing new shoes is essential in the huge shoe industry. 

At first glance, looking through the formatted and labeled datasets already available on the web, none stand out that have only shoes. One possibility would be to extract from these datasets only the images that we want.
Another possibility would be to scrap, for example, some shoe stores' website such as \textit{zalando.be} or \textit{Asos.com} and gather the images with their labels. This has as an advantage that we don't have to do the labelling ourselves.

## Nices-to-have
Moving further, we would like to explore the idea of using text parameters to generate the shoes of our liking. For example, imagine the possibility of adding the colour as input to our network, the network then outputting and image of a completely new shoe having the specified colour. Another parameter could be the shoe type, for example sneaker or boots.

Going even further, we could also extend this idea of text to image to an image to image generation, where the first image determines the pattern of the second. For example, one could imagine that feeding a leopard bag into the network would output a shoe with a leopard skin pattern.
