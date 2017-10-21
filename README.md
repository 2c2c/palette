# palette
A color quantization script using k-means clustering. It finds the best `n` colors that make up an image and redraws them along with the palette that made it possible.

Using Lab colorspace throughout the calculations lead to big gains in accuracy. 

![papika](https://i.imgur.com/h1qQY3i.jpg)

![bird](https://i.imgur.com/GwHwP6I.jpg)

![palm2](https://i.imgur.com/w0JuWSw.jpg)

It can render multiple iterations increasing the number of colors.

![palm](https://i.imgur.com/F06w9TA.jpg)

![girl](https://i.imgur.com/ZbFIFA1.jpg)

![china](https://i.imgur.com/bQ599XA.png)

To use: `python palette.py imagename startcolors endcolors`