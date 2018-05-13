import argparse
import time

import numpy as np
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is a segmentation program using mean-shift algorithm')
    parser.add_argument("filename")
    parser.add_argument("bandwidth", type=int)
    parser.add_argument("--stripe", type=int, default=40)
    parser.add_argument("--gaussian", action="store_true")
    parser.add_argument("--iterations", action="store", default=15)
    args = parser.parse_args()
    # print(args)
    filename = args.filename
    bandwidth = args.bandwidth
    stripe = args.stripe
    gaussian = args.gaussian
    iterations = args.iterations
    m = 1
    S = 5
    threshold = 1.0
    img = Image.open(filename)
    img = np.array(img)
    segmented_image = img.copy()
    rows, columns, dim = img.shape

    start_time = time.time()
    print("mean shift start")
    means = []
    for row in range(0, rows, stripe):
        for column in range(0, columns, stripe):
            now_row = min(int(row + np.random.random() * stripe), rows - 1)
            now_column = min(int(column + np.random.random() * stripe), columns - 1)
            now = np.array([now_row, now_column, *img[now_row, now_column]])
            for iteration in range(iterations):
                x = now[0]
                y = now[1]
                r1 = max(0, x - stripe)
                r2 = min(x + stripe, rows)
                c1 = max(0, y - stripe)
                c2 = min(y + stripe, columns)
                kernel = []
                for i in range(r1, r2):
                    for j in range(c1, c2):
                        dc = np.linalg.norm(img[i][j] - now[2:])
                        ds = (np.linalg.norm(np.array([i, j]) - now[:2])) * m / S
                        D = np.linalg.norm([dc, ds])
                        if D < bandwidth:
                            kernel.append([i, j, *img[i][j]])
                kernel = np.array(kernel)
                if not gaussian:
                    mean = np.mean(kernel, axis=0, dtype=np.int32)

                dc = np.linalg.norm(now[2:] - mean[2:])
                ds = (np.linalg.norm(now[:2] - mean[:2])) * m / S
                dsm = np.linalg.norm([dc, ds])
                now = mean
                if dsm <= threshold:
                    break
            means.append(now)
    end_time = time.time()
    print("mean shift end for %.1f s" % (end_time - start_time))

    flags = [True for _ in means]
    means = np.array(means, dtype=np.float32)
    print(means)
    converged_means = []

    for i, mean in enumerate(means):
        if flags[i]:
            w = 1.0
            for j in range(i + 1, len(means)):
                dc = np.linalg.norm(means[i][2:] - means[j][2:])
                ds = (np.linalg.norm(means[i][:2] - means[j][:2])) * m / S
                dsm = np.linalg.norm([dc, ds])
                if dsm < bandwidth:
                    means[i] += means[j]
                    w += 1.0
                    flags[j] = 0
            means[i] /= w
            converged_means.append(means[i])
    converged_means = np.array(converged_means)
    print("means converged")

    min_dist = np.zeros(shape=[rows, columns]) + 1e10
    labels = np.zeros(shape=[rows, columns], dtype=np.int32) - 1
    for i in range(rows):
        for j in range(columns):
            for c in range(len(converged_means)):
                dc = np.linalg.norm(img[i][j] - converged_means[c][2:])
                ds = (np.linalg.norm(np.array([i, j]) - converged_means[c][:2])) * m / S
                D = np.linalg.norm([dc, ds])
                if D < min_dist[i][j]:
                    min_dist[i][j] = D
                    labels[i][j] = c
            segmented_image[i][j] = converged_means[labels[i][j]][2:]
    segmented_image = Image.fromarray(segmented_image)
    segmented_image.save("%s_output_%s_%d.jpg" % (("gaussian" if gaussian else "uniform"), filename, bandwidth))
