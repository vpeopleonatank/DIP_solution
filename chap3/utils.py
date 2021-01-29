import matplotlib.pyplot as plt


def display_multi_img(imgs):
    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    columns = len(imgs)
    rows = 1
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(imgs[i - 1], cmap="gray")
    plt.show()
