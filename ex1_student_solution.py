"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple

from numpy.linalg import svd
from scipy.interpolate import griddata

PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""

    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        # return homography
        number_of_points = match_p_src.shape[1]
        src_coordinate_mat = np.concatenate((match_p_src, np.ones((1, number_of_points))), axis=0)
        A = np.vstack((
            [Solution.create_matching_coordinate_pair_rows(index, coordinate_pair, match_p_dst)
             for index, coordinate_pair in enumerate(src_coordinate_mat.T)]
        ))

        [_, _, V_t] = np.linalg.svd(A)
        return V_t[-1].reshape(3, 3) / V_t[-1, -1]

    @staticmethod
    def create_matching_coordinate_pair_rows(index: int,
                                             src_coordinate_vector: np.ndarray,
                                             dst_coordinate_mat: np.ndarray) -> np.ndarray:
        return np.stack((
            np.r_[src_coordinate_vector, [0] * 3, -1 * dst_coordinate_mat[0][index] * src_coordinate_vector],
            np.r_[[0] * 3, src_coordinate_vector, -1 * dst_coordinate_mat[1][index] * src_coordinate_vector]))

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        h, w = src_image.shape[:2]
        dst_image = np.zeros(dst_image_shape, dtype=np.uint8)
        for src_y in range(h):
            for src_x in range(w):
                dst_coordinates = homography @ np.array([src_x, src_y, 1])
                dst_x, dst_y = np.round(dst_coordinates[:2] / dst_coordinates[2]).astype(int)
                # there is a flip in the coordinates, as (row, column) = (y, x)
                if 0 <= dst_y < dst_image_shape[0] and 0 <= dst_x < dst_image_shape[1]:
                    dst_image[dst_y][dst_x] = src_image[src_y][src_x]

        return dst_image

    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        dst_image = np.zeros(dst_image_shape, dtype=np.uint8)
        src_h, src_w = src_image.shape[:2]
        dst_h, dst_w = dst_image_shape[:2]
        src_yy, src_xx = np.meshgrid(range(src_h), range(src_w))
        src_yy, src_xx = src_yy.ravel(), src_xx.ravel()
        src_coordinate_matrix = np.array([src_xx, src_yy, np.ones((src_h * src_w))], dtype=np.int32)
        dst_coordinate_matrix = homography @ src_coordinate_matrix
        dst_coordinate_matrix = np.round(dst_coordinate_matrix[:2] / dst_coordinate_matrix[2]).astype(int)
        valid_dst_xx, valid_dst_yy, valid_src_xx, valid_src_yy = \
            Solution.get_valid_indices(dst_coordinate_matrix, dst_h, dst_w, src_xx, src_yy)

        dst_image[valid_dst_yy, valid_dst_xx] = src_image[valid_src_yy, valid_src_xx]
        return dst_image

    @staticmethod
    def get_valid_indices(dst_coordinate_matrix, dst_h, dst_w, src_xx, src_yy):
        valid_dst_indices = np.flatnonzero((0 <= dst_coordinate_matrix[0]) & (dst_coordinate_matrix[0] < dst_w)
                                           & (0 <= dst_coordinate_matrix[1]) & (dst_coordinate_matrix[1] < dst_h))

        valid_dst_xx, valid_dst_yy = dst_coordinate_matrix[:, valid_dst_indices]
        valid_src_xx, valid_src_yy = src_xx[valid_dst_indices], src_yy[valid_dst_indices]
        return valid_dst_xx, valid_dst_yy, valid_src_xx, valid_src_yy

    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        distances = Solution.get_matching_point_distances(homography, match_p_dst, match_p_src)
        inlier_indices = distances < max_err

        fit_percent = np.mean(inlier_indices)
        dist_mse = np.mean(distances[inlier_indices]) if np.any(inlier_indices) else 10 ** 9
        return fit_percent, dist_mse

    @staticmethod
    def get_matching_point_distances(homography: np.ndarray,
                                     match_p_dst: np.ndarray,
                                     match_p_src: np.ndarray) -> np.ndarray:
        number_of_points = match_p_src.shape[1]
        src_coordinate_mat = np.concatenate((match_p_src, np.ones((1, number_of_points), dtype=int)), axis=0)
        transformed_coordinate_mat = homography @ src_coordinate_mat
        transformed_coordinate_mat = transformed_coordinate_mat[:2] / transformed_coordinate_mat[2]
        distances = np.linalg.norm(match_p_dst - transformed_coordinate_mat, axis=0)
        return distances

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        distances = Solution.get_matching_point_distances(homography, match_p_dst, match_p_src)
        inlier_indices = distances < max_err

        return match_p_src[:, inlier_indices], match_p_dst[:, inlier_indices]

    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # # use class notations:
        # w = inliers_percent
        # # t = max_err
        # # p = parameter determining the probability of the algorithm to
        # # succeed
        # p = 0.99
        # # the minimal probability of points which meets with the model
        # d = 0.5
        # # number of points sufficient to compute the model
        # n = 4
        # # number of RANSAC iterations (+1 to avoid the case where w=1)
        # k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1
        w = inliers_percent
        t = max_err
        p = 0.99
        d = 0.5
        n = 4
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1

        best_model, best_mse = np.empty(np.array([3, 3])), np.finfo(float).max
        for i in range(k):
            candidate_indices = np.random.permutation(range(match_p_src.shape[1]))[:n]
            src_candidates, dst_candidates = match_p_src[:, candidate_indices], match_p_dst[:, candidate_indices]
            candidate_homography = Solution.compute_homography_naive(src_candidates, dst_candidates)
            fit_percent, dist_mse = Solution.test_homography(candidate_homography, match_p_src, match_p_dst, t)

            if fit_percent > d:
                src_inliers, dst_inliers = Solution.meet_the_model_points(candidate_homography, match_p_src,
                                                                          match_p_dst, t)

                candidate_homography = Solution.compute_homography_naive(src_inliers, dst_inliers)
                fit_percent, dist_mse = Solution.test_homography(candidate_homography, match_p_src, match_p_dst, t)
                if dist_mse < best_mse:
                    best_mse, best_model = dist_mse, candidate_homography

        return best_model

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """
        backward_warp = np.zeros(dst_image_shape, dtype=np.uint8)

        dst_h, dst_w = dst_image_shape[:2]
        src_h, src_w = src_image.shape[:2]
        dst_yy, dst_xx = np.meshgrid(range(dst_h), range(dst_w))
        dst_coordinate_matrix = np.array([dst_xx.ravel(), dst_yy.ravel(), np.ones((dst_h * dst_w))], dtype=np.int32)
        transformed_coordinate_matrix = backward_projective_homography @ dst_coordinate_matrix
        transformed_coordinate_matrix = transformed_coordinate_matrix[:2] / transformed_coordinate_matrix[2]
        src_yy, src_xx = np.meshgrid(range(src_h), range(src_w))
        src_coordinates = np.array([src_xx.ravel(), src_yy.ravel()], dtype=np.int32)
        src_values = src_image.T.reshape(3, -1)
        grid = griddata(src_coordinates.T, src_values.T, transformed_coordinate_matrix.T, method='cubic', fill_value=0)

        backward_warp = grid.reshape((dst_h, dst_w, 3), order='F')
        return backward_warp

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        # forward mapped source coordinates need to be shifted (pad_left, pad_up) after the mapping (matrix
        # multiplication from the left), so backward mapping gets reverse shift matrix multiplied from the right
        translation_matrix = np.array([[1, 0, -pad_left], [0, 1, -pad_up], [0, 0, 1]])
        final_homography = backward_homography @ translation_matrix
        return final_homography / final_homography[2, 2]

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        forward_homography = self.compute_homography(match_p_src, match_p_dst, inliers_percent, max_err)
        pano_rows, pano_cols, pad_struct = Solution.find_panorama_shape(src_image, dst_image, forward_homography)
        dst_up, dst_down = pad_struct.pad_up, pad_struct.pad_up + dst_image.shape[0]
        dst_left, dst_right = pad_struct.pad_left, pad_struct.pad_left + dst_image.shape[1]
        backward_homography = np.linalg.inv(forward_homography)
        backward_homography = Solution.add_translation_to_backward_homography(backward_homography, dst_left, dst_up)
        mapped_src = Solution.compute_backward_mapping(backward_homography, src_image, (pano_rows, pano_cols, 3))
        # we override the panorama matrix with destination values, so initializing with mapped source image is easier
        panorama = mapped_src
        panorama[dst_up:dst_down, dst_left:dst_right, :] = dst_image

        return np.clip(panorama, 0, 255).astype(np.uint8)
