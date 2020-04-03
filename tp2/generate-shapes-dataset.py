from PIL import Image, ImageDraw
import random

# generate 100 shapes of each type: rectangles, ellipses, circles, squares and triangles
# draw rectangles
for i in range(100):
    image = Image.new('RGB', (64, 64), 'white')  # could also open an existing image here to draw shapes over it
    draw = ImageDraw.Draw(image)
    x0 = random.randint(1,30)
    y0 = random.randint(1,30)
    x1 = random.randint(35,63)
    y1 = random.randint(35,63)
    draw.rectangle((x0, y0, x1, y1), fill='red', outline='red')  # can vary this bit to draw different shapes in different positions
    image.save('./rectangle/rectangle' + str(i) + '.png')

# draw ellipse
for i in range(100):
    image = Image.new('RGB', (64, 64), 'white')  # could also open an existing image here to draw shapes over it
    draw = ImageDraw.Draw(image)
    x0 = random.randint(1,30)
    y0 = random.randint(1,30)
    x1 = random.randint(35,63)
    y1 = random.randint(35,63)
    draw.ellipse((x0, y0, x1, y1), fill='red', outline='red')  # can vary this bit to draw different shapes in different positions
    image.save('./ellipse/ellipse' + str(i) + '.png')

# draw circle
for i in range(100):
    image = Image.new('RGB', (64, 64), 'white')  # could also open an existing image here to draw shapes over it
    draw = ImageDraw.Draw(image)
    x0 = random.randint(1,30)
    y0 = random.randint(1,30)
    dif = random.randint(4,20)
    x1 = x0 + dif
    y1 = y0 + dif
    draw.ellipse((x0, y0, x1, y1), fill='red', outline='red')  # can vary this bit to draw different shapes in different positions
    image.save('./circle/circle' + str(i) + '.png')

# draw square
for i in range(100):
    image = Image.new('RGB', (64, 64), 'white')  # could also open an existing image here to draw shapes over it
    draw = ImageDraw.Draw(image)
    x0 = random.randint(1,30)
    y0 = random.randint(1,30)
    dif = random.randint(4,20)
    x1 = x0 + dif
    y1 = y0 + dif
    draw.rectangle((x0, y0, x1, y1), fill='red', outline='red')  # can vary this bit to draw different shapes in different positions
    image.save('./square/square' + str(i) + '.png')

# draw triangle
for i in range(100):
    image = Image.new('RGB', (64, 64), 'white')  # could also open an existing image here to draw shapes over it
    draw = ImageDraw.Draw(image)
    x0 = random.randint(1,63)
    y0 = random.randint(1,63)
    x1 = random.randint(1,63)
    y1 = random.randint(1,63)
    x2 = random.randint(1,63)
    y2 = random.randint(1,63)
    draw.polygon([(x0, y0), (x1, y1), (x2, y2)], fill='red', outline='red')  # can vary this bit to draw different shapes in different positions
    image.save('./triangle/triangle' + str(i) + '.png')
