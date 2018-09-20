import numpy as np
from matplotlib import pyplot as plt
import pathlib

import logging
import timeit
import os
import shutil

from k_means import k_means

logger = logging.getLogger(__name__)


class TestKMeansPerfomance:
    MIN_DOTS = 100
    MAX_DOTS = 2000
    STEP_DOTS = 100

    MIN_FEATURES = 1
    MAX_FEATURES = 10
    STEP_FEATURES = 1

    MIN_CENTERS = 1
    MAX_CENTERS = 20
    STEP_CENTERS = 1

    REPEATED = 10
    NUM_ITER = 10
    PATH_FOR_SAVING = os.getcwd()

    def run_perfomance_test(self, path: pathlib.Path = None):
        if path is None:
            path = pathlib.Path(self.PATH_FOR_SAVING)

        # self._test_growing_num_centroids_with_fixed_num_features_and_num_dots(path)
        self._test_growing_num_dots_with_fixed_num_features_and_num_centroids(path)

    def _test_growing_num_dots_with_fixed_num_features_and_num_centroids(self, path: pathlib.Path):
        dir_path = self._prepate_folder_for_saving_results(path, "test_growing_dots")

        dots_for_testing = self._produce_dots_fot_testing(
            self.MIN_DOTS,
            self.MAX_DOTS,
            self.STEP_DOTS,
            self.MAX_FEATURES
        )

        logger.info("tested dots is generated")

        centroids = self.get_dots(self.MAX_CENTERS, self.MAX_FEATURES)

        time_exec_s = []
        time_exec_p = []
        dots_num = []

        for dots, num_dots in dots_for_testing:
            logger.info(f"START calculation for NUM_DOTS = {num_dots}")
            time_p = timeit.timeit(lambda: k_means.kmeans_parallel(
                dots,
                self.MAX_CENTERS,
                self.NUM_ITER,
                num_dots,
                self.MAX_FEATURES,
                centroids

            ), number=self.REPEATED)

            time_s = timeit.timeit(lambda: k_means.kmeans_straight(
                dots,
                self.MAX_CENTERS,
                self.NUM_ITER,
                num_dots,
                self.MAX_FEATURES,
                centroids

            ), number=self.REPEATED)

            logger.info(f"TIME of calculation for NUM_DOTS PARALLEL = {time_p}")
            logger.info(f"TIME of calculation for NUM_DOTS STRAIGHT = {time_s}")

            time_exec_s.append(time_s)
            time_exec_p.append(time_p)
            dots_num.append(num_dots)

        plot_path_prl = dir_path.joinpath("parallel_dependency.jpg")
        plot_path_straight = dir_path.joinpath("straight_dependency.jpg")

        plt.plot(dots_num, time_exec_s)
        plt.savefig(plot_path_straight)
        plt.close()

        logger.info(f"Saving graphic of plt by PATH f{plot_path_straight}")

        plt.plot(dots_num, time_exec_p)
        plt.savefig(plot_path_prl)
        plt.close()

        logger.info(f"Saving graphic of plt by PATH f{plot_path_prl}")

    def _test_growing_num_centroids_with_fixed_num_features_and_num_dots(self, path: pathlib.Path):
        dir_path = self._prepate_folder_for_saving_results(path, "test_growing_centroids")

        centers_for_testing = self._produce_centroids_fot_testing(
            self.MIN_CENTERS,
            self.MAX_CENTERS,
            self.STEP_CENTERS,
            self.MAX_FEATURES
        )
        logger.info("tested centers is generated")

        dots = self.get_dots(self.MAX_DOTS, self.MAX_FEATURES)

        time_exec_s = []
        time_exec_p = []
        centroids_num = []

        for centroids, num_centroids in centers_for_testing:
            logger.info(f"START calculation for NUM_CENTROIDS = {num_centroids}")

            time_s = timeit.timeit(
                lambda: k_means.kmeans_straight(
                    dots,
                    num_centroids,
                    self.NUM_ITER,
                    self.MAX_DOTS,
                    self.MAX_FEATURES,
                    centroids

                ), number=self.REPEATED)

            time_p = timeit.timeit(
                lambda: k_means.kmeans_parallel(
                    dots,
                    num_centroids,
                    self.NUM_ITER,
                    self.MAX_DOTS,
                    self.MAX_FEATURES,
                    centroids

                ), number=self.REPEATED)

            logger.info(f"TIME of calculation for NUM_CENTROIDS PARALLEL = {time_p}")
            logger.info(f"TIME of calculation for NUM_CENTROIDS STRAIGHT = {time_s}")

            time_exec_s.append(time_s)
            time_exec_p.append(time_p)

            centroids_num.append(num_centroids)

        plot_path_prl = dir_path.joinpath("parallel_dependency.jpg")
        plot_path_straight = dir_path.joinpath("straight_dependency.jpg")

        plt.plot(centroids_num, time_exec_s)
        plt.savefig(plot_path_straight)
        plt.close()

        logger.info(f"Saving graphic of plt by PATH f{plot_path_straight}")

        plt.plot(centroids_num, time_exec_p)
        plt.savefig(plot_path_prl)
        plt.close()

        logger.info(f"Saving graphic of plt by PATH f{plot_path_prl}")

    def _produce_dots_fot_testing(self, start_num_dots, max_num_dots, step, num_features):
        for num_dots in range(start_num_dots, max_num_dots, step):
            yield self.get_dots(num_dots, num_features), num_dots

    def _produce_centroids_fot_testing(self, start_num_centroids, max_num_centroids, step, num_features):
        for num_centroids in range(start_num_centroids, max_num_centroids, step):
            yield self.get_dots(num_centroids, num_features), num_centroids

    def get_dots(self, num, features):
        return np.random.ranf((num, features))

    def _prepate_folder_for_saving_results(self, path: pathlib.Path, dir_name: str):
        dir_path = path.joinpath(dir_name)

        if dir_path.exists():
            shutil.rmtree(dir_path)

        dir_path.mkdir()

        logger.info("dir for result is created")

        return dir_path
