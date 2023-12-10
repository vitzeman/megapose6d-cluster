import os
import json
from typing import Tuple, Union

import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle


class TransformationVisualizer:
    def __init__(
        self,
        xlim: tuple = (-1, 1),
        ylim: tuple = (-1, 1),
        zlim: tuple = (-1, 1),
        units: str = "-",
    ) -> None:
        self.fig = plt.figure(figsize=(7, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)

        self.ax.set_xlabel(f"x [{units}]")
        self.ax.set_ylabel(f"y [{units}]")
        self.ax.set_zlabel(f"z [{units}]")

        self.length = max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]) * 0.1

        self.ax.quiver(0, 0, 0, 1, 0, 0, length=self.length, color="r", linewidth=1)
        self.ax.quiver(0, 0, 0, 0, 1, 0, length=self.length, color="g", linewidth=1)
        self.ax.quiver(0, 0, 0, 0, 0, 1, length=self.length, color="b", linewidth=1)
        self.ax.text(0, 0, 0, "O", fontweight="bold")

        # Points of the coordinate system in homogeneous coordinates
        self.O = np.array([0, 0, 0, 1]).reshape(4, 1)
        self.X = np.array([1, 0, 0, 1]).reshape(4, 1)
        self.Y = np.array([0, 1, 0, 1]).reshape(4, 1)
        self.Z = np.array([0, 0, 1, 1]).reshape(4, 1)

    def add_transformation(
        self,
        T_mtx: np.ndarray,
        text: str = None,
    ) -> None:
        if isinstance(T_mtx, R):
            T_mtx = T_mtx.as_matrix()

        assert T_mtx.shape == (4, 4), "Transformation matrix must be 4x4"

        # Transform the points of the coordinate system
        O = T_mtx @ self.O  # Origin
        X = T_mtx @ self.X
        Y = T_mtx @ self.Y
        Z = T_mtx @ self.Z

        O = O[:3].reshape(3)
        X = X[:3].reshape(3)
        Y = Y[:3].reshape(3)
        Z = Z[:3].reshape(3)

        dx = X - O
        dy = Y - O
        dz = Z - O

        check = np.dot(dx, dy)
        assert np.isclose(check, 0), "The vectors dx and dy are not orthogonal"
        check = np.dot(dx, dz)
        assert np.isclose(check, 0), "The vectors dx and dz are not orthogonal"
        check = np.dot(dy, dz)
        assert np.isclose(check, 0), "The vectors dy and dz are not orthogonal"

        self.ax.quiver(
            O[0], O[1], O[2], dx[0], dx[1], dx[2], length=self.length, color="r", linewidth=1
        )
        self.ax.quiver(
            O[0], O[1], O[2], dy[0], dy[1], dy[2], length=self.length, color="g", linewidth=1
        )
        self.ax.quiver(
            O[0], O[1], O[2], dz[0], dz[1], dz[2], length=self.length, color="b", linewidth=1
        )

        if text is not None:
            self.ax.text(O[0], O[1], O[2], text, fontweight="bold")

    def add_title(self, title: str) -> None:
        self.ax.set_title(title)

    def show(self) -> None:
        self.ax.legend()
        plt.show()

    def save(self, path: str) -> None:
        self.ax.legend()
        plt.savefig(path)

    def add_point(self, point: np.ndarray, color: str = "k", text: str = None) -> None:
        self.ax.scatter(point[0], point[1], point[2], color=color)
        if text is not None:
            self.ax.text(point[0], point[1], point[2], text, fontweight="bold")


if __name__ == "__main__":
    with open(
        os.path.join("6D_pose_dataset", "BOP_format", "Capture", "8", "transforms_all.json"), "r"
    ) as f:
        transforms = json.load(f)

    pose_vis = TransformationVisualizer()
    for frame in tqdm(transforms["frames"]):
        T = np.array(frame["transform_matrix"])
        print(T)
        T = T @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pose_vis.add_transformation(T)
    pose_vis.show()
