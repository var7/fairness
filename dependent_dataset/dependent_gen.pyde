size(500, 500)
# for valid random seed = 2018
# for test random seed = 1994
# for train random seed = 1791387
randomSeed(1994)
num_imgs = 500

img_size = 500
min_object_size = 30
max_object_size = 300
num_objects = 1

# shapes = zeros((num_imgs, num_objects), dtype=int)
num_shapes = 2
shape_labels = ['rectangle', 'circle']
# colors = zeros((num_imgs, num_objects), dtype=int)
num_colors = 2
color_labels = ['red', 'green']
num_lines = 5
colors = []
shapes = []
for i_img in range(num_imgs):
    clear()
    r_bg = (random(0, 150))
    g_bg = (random(0, 150))
    b_bg = (random(0, 255))
    background(r_bg, g_bg, b_bg)

    for i_object in range(num_objects):
        
        shape = int(random(num_shapes))
        shapes.append(shape)
        w = int(random(min_object_size, max_object_size))
        h = int(random(min_object_size, max_object_size))
        
        # color = int(random(0, num_colors))
        max_offset = 0.2
        r_offset = max_offset * 2. * (random(3) - 0.5)
        g_offset = max_offset * 2. * (random(3) - 0.5)
        b_offset = max_offset * 2. * (random(3) - 0.5)
        alpha = random(200, 255)
        
        if shape == 0:
            prob_red = random(1)
            if prob_red < 0.8:
                colr = 0
                fill(255 - max_offset + r_offset, 0 + g_offset, 0 + b_offset, alpha)
            else:
                colr = 1
                fill(0 + r_offset, 255 - max_offset + g_offset, 0 + b_offset, alpha)  
        else:
            prob_green = random(1)
            if prob_green < 0.8:
                colr = 1
                fill(0 + r_offset, 255 - max_offset + g_offset, 0 + b_offset, alpha)
            else:
                colr = 0
                fill(255 - max_offset + r_offset, 0 + g_offset, 0 + b_offset, alpha)
                
        colors.append(colr)
        noStroke()
        # shapes[i_img, i_object] = shape
        if shape == 0:  # rectangle
            x = int(random(0, img_size - w))
            y = int(random(0, img_size - h))
            rect(x, y, w, h)
        elif shape == 1:  # circle
            x = int(random(w, img_size - w))
            y = int(random(h, img_size - h))
            ellipse(x, y, w, w)

            # TODO: Introduce some variation to the colors by adding a small
            # random offset to the rgb values.

    for i_line in range(num_lines):
        drawLine = random(1)
        if drawLine < 0.3:
            continue
        choice = random(1)
        if choice > 0.5:
            x1 = img_size + 1
            y1 = int(random(img_size))
            x2 = 0
            y2 = int(random(img_size))
        else:
            x1 = int(random(img_size))
            y1 = img_size + 1
            x2 = int(random(img_size))
            y2 = 0
        offset = random(1)
        stroke((255 - r_bg) * offset, (255 - g_bg)
               * offset, (255 - b_bg) * offset)
        line(x1, y1, x2, y2)

    save('./test/{}/{}_{}.png'.format(shape_labels[shape], i_img, color_labels[colr]))

file = createWriter("./test_dataset.txt")
for ind, (sh, col) in enumerate(zip(shapes, colors)):
    # Write the datum to the file
    file.print('{}\t{}\t{}\n'.format(ind, shape_labels[sh], color_labels[col]))
file.flush()  # Writes the remaining data to the file
file.close()  # Finishes the file

# shapes_file = createWriter("./shapes.txt")
# for shape in shapes:
# shapes_file.print('{}\n'.format(shape)) # Write the datum to the file
# shapes_file.flush()# Writes the remaining data to the file
# shapes_file.close()# Finishes the file
exit()
