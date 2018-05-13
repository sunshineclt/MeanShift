import argparse
import multiprocessing as mp
import time

import numpy as np
from PIL import Image

from kernel import gaussian_mean


def mean_shift(row, column):
    now_row = min(int(row + np.random.random() * stripe), rows - 1)
    now_column = min(int(column + np.random.random() * stripe), columns - 1)
    now = np.array([now_row, now_column, *img[now_row, now_column]])
    for _ in range(iterations):
        x = now[0]
        y = now[1]
        r1 = max(0, x - stripe)
        r2 = min(x + stripe, rows)
        c1 = max(0, y - stripe)
        c2 = min(y + stripe, columns)
        kernel_points = []
        for i in range(r1, r2):
            for j in range(c1, c2):
                dc = np.linalg.norm(img[i][j] - now[2:])
                ds = (np.linalg.norm(np.array([i, j]) - now[:2])) * m / S
                D = np.linalg.norm([dc, ds])
                if D < bandwidth:
                    kernel_points.append([i, j, *img[i][j]])
        kernel_points = np.array(kernel_points)
        if gaussian:
            mean = gaussian_mean(kernel_points, now, bandwidth)
        else:
            mean = np.mean(kernel_points, axis=0, dtype=np.int32)

        dc = np.linalg.norm(now[2:] - mean[2:])
        ds = (np.linalg.norm(now[:2] - mean[:2])) * m / S
        dsm = np.linalg.norm([dc, ds])
        now = mean
        if dsm <= threshold:
            break
    return now


def draw_segmented(row, column):
    min_dist = 1e10
    label = -1
    for c in range(len(converged_means)):
        dc = np.linalg.norm(img[row][column] - converged_means[c][2:])
        ds = (np.linalg.norm(np.array([row, column]) - converged_means[c][:2])) * m / S
        D = np.linalg.norm([dc, ds])
        if D < min_dist:
            min_dist = D
            label = c
    return row, column, converged_means[label][2:]


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
    print("%s segmentation with bandwidth %d, %s kernel" % (filename, bandwidth, ("gaussian" if gaussian else "uniform")))
    m = 1
    S = 5
    threshold = 1.0
    img = Image.open(filename)
    img = np.array(img)
    segmented_image = img.copy()
    rows, columns, dim = img.shape

    # Set up for multi-processing
    pool = mp.Pool(processes=12)

    # Mean Shift
    condition = []
    for row in range(0, rows, stripe):
        for column in range(0, columns, stripe):
            condition.append((row, column))
    start_time = time.time()
    print("mean shift start")
    means = pool.starmap(mean_shift, condition)
    end_time = time.time()
    print("mean shift end for %.1f s" % (end_time - start_time))
    means = np.array(means, dtype=np.float32)

    # Converge Means
    flags = [True for _ in means]
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
    print("number of converged means is ", len(converged_means))
    print("means converged")

    # Draw Segmented Image
    pool = mp.Pool(processes=12)
    condition = []
    for row in range(0, rows):
        for column in range(0, columns):
            condition.append((row, column))
    result = pool.starmap(draw_segmented, condition)
    for i in result:
        segmented_image[i[0]][i[1]] = i[2]
    segmented_image = Image.fromarray(segmented_image)
    segmented_image.save("%s_output_%s_%d.jpg" % (("gaussian" if gaussian else "uniform"), filename, bandwidth))
    end_time = time.time()
    print("total time is %.1f s" % (end_time - start_time))
