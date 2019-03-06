# Shoes-generator

## Project description
Our main idea for this project is to use deep learning techniques to generate new unseen images from a model that has been trained on an image dataset.
For the scale of this project however, it is wise to restrict ourselves to one type of object rather than a multitude of different ones. Our preference lies on a shoe dataset, as the process of designing new shoes is essential in the huge shoe industry. 

At first glance, looking through the formatted and labeled datasets already available on the web, none stand out that have only shoes. One possibility would be to extract from these datasets only the images that we want.
Another possibility would be to scrap, for example, some shoe stores' website such as \textit{zalando.be} or \textit{Asos.com} and gather the images with their labels. This has as an advantage that we don't have to do the labelling ourselves.

## Nices-to-have
Moving further, we would like to explore the idea of using text parameters to generate the shoes of our liking. For example, imagine the possibility of adding the colour as input to our network, the network then outputting and image of a completely new shoe having the specified colour. Another parameter could be the shoe type, for example sneaker or boots.

Going even further, we could also extend this idea of text to image to an image to image generation, where the first image determines the pattern of the second. For example, one could imagine that feeding a leopard bag into the network would output a shoe with a leopard skin pattern.
