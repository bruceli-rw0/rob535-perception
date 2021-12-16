import numpy as np
import matplotlib.pyplot as plt

def visualize_bbox_yolo(img: np.ndarray, annotation: list):
    vertex = [
        [int(annotation[1] - annotation[3] / 2), int(annotation[2] - annotation[4] / 2)],
        [int(annotation[1] + annotation[3] / 2), int(annotation[2] - annotation[4] / 2)],
        [int(annotation[1] + annotation[3] / 2), int(annotation[2] + annotation[4] / 2)],
        [int(annotation[1] - annotation[3] / 2), int(annotation[2] + annotation[4] / 2)],
    ]

    fig1 = plt.figure(1, figsize=(16, 9))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.imshow(img)
    ax1.axis('scaled')
    fig1.tight_layout()

    vertex_ = vertex[-1:] + vertex[:-1]
    for pt1, pt2 in zip(vertex, vertex_):    
        ax1.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='r')

#------------------------------Given by ROB staff------------------------------#

def rot(n):
    n = np.asarray(n).flatten()
    assert(n.size == 3)

    theta = np.linalg.norm(n)
    if theta:
        n /= theta
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

        return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    else:
        return np.identity(3)


def get_bbox(p0, p1):
    """
    Input:
    *   p0, p1
        (3)
        Corners of a bounding box represented in the body frame.

    Output:
    *   v
        (3, 8)
        Vertices of the bounding box represented in the body frame.
    *   e
        (2, 14)
        Edges of the bounding box. The first 2 edges indicate the `front` side
        of the box.
    """
    v = np.array([
        [p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
        [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
        [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]
    ])
    e = np.array([
        [2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
        [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]
    ], dtype=np.uint8)

    return v, e
