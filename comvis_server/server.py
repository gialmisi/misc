import socket
import sys

import numpy as np
import pandas as pd
from desdeo_emo.EAs.RVEA import RVEA
from desdeo_problem.Objective import _ScalarObjective
from desdeo_problem.Problem import MOProblem
from desdeo_problem.Variable import variable_builder
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.mixture import GaussianMixture

from myparser import parse_dict, parse_message
from solution_color import color_solutions

n_clusters = 5

# create the problem
def f_1(x):
    # max
    res = 4.07 + 2.27 * x[:, 0]
    return res


def f_2(x):
    # max
    res = 2.60 + 0.03 * x[:, 0] + 0.02 * x[:, 1] + 0.01 / (1.39 - x[:, 0] ** 2) + 0.30 / (1.39 - x[:, 1] ** 2)
    return res


def f_3(x):
    # max
    res = 8.21 - 0.71 / (1.09 - x[:, 0] ** 2)
    return res


def f_4(x):
    # min
    res = -0.96 + 0.96 / (1.09 - x[:, 1] ** 2)
    return res


# def f_5(x):
# return -0.96 + 0.96 / (1.09 - x[:, 1]**2)


def f_5(x):
    # min
    return np.max([np.abs(x[:, 0] - 0.65), np.abs(x[:, 1] - 0.65)], axis=0)


f1 = _ScalarObjective(name="f1", evaluator=f_1, maximize=[True])
f2 = _ScalarObjective(name="f2", evaluator=f_2, maximize=[True])
f3 = _ScalarObjective(name="f3", evaluator=f_3, maximize=[True])
f4 = _ScalarObjective(name="f4", evaluator=f_4)
f5 = _ScalarObjective(name="f5", evaluator=f_5)

varsl = variable_builder(["x_1", "x_2"], initial_values=[0.5, 0.5], lower_bounds=[0.3, 0.3], upper_bounds=[1.0, 1.0],)

problem = MOProblem(variables=varsl, objectives=[f1, f2, f3, f4, f5])

evolver = RVEA(problem, interact=True, n_iterations=10, n_gen_per_iter=100)

_, pref = evolver.iterate()
n_iteration = 1

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("10.0.2.15", 5005))
# sock.bind(("127.0.0.1", 5005))
sock.listen(1)

print("Waiting for ComVis to connect...")
connection, client_addr = sock.accept()

print("Connection estabilished!")

while True:
    try:
        data = connection.recv(2048)
        print(f"Recieved: {data}")

        if not data:
            print("Connection lost!")
            # reset the evolver
            evolver = RVEA(problem, interact=True, n_iterations=10, n_gen_per_iter=100)
            _, pref = evolver.iterate()
            n_iteration = 1
            connection.close()

            sock.listen(1)
            print("Waiting for ComVis to connect...")
            connection, client_addr = sock.accept()
            print("Connection estabilished!")

        else:
            # do stuff with data
            d = parse_message(data.decode("utf-8"))
            print(f"Parsed message: {d}")

            if d["DATA"] == "":
                # No preference is sent. Sending current data back.
                color_point = evolver.population.ideal_objective_vector
            else:
                # Preference sent. Run one iteration of evolver
                ref_point = np.squeeze(eval(d["DATA"]))
                print(f"Ref point: {ref_point}")
                pref.response = pd.DataFrame(np.atleast_2d(ref_point), columns=pref.content["dimensions_data"].columns,)
                color_point = pref.response.values
                _, pref = evolver.iterate(pref)

            objectives_ = evolver.population.objectives

            ### KMEANS
            # fit to n_clusters and find the closest solutions to each cluster's centroid
            kmeans = KMeans(n_clusters=n_clusters, verbose=0)
            kmeans.fit(objectives_)
            closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, objectives_)
            labels_kmeans = kmeans.labels_
            print(labels_kmeans)

            labelled_objectives = []
            labelled_variables = []
            for label_n in range(n_clusters):
                labelled_objectives.append(objectives_[labels_kmeans == label_n])
                labelled_variables.append(evolver.population.individuals[labels_kmeans == label_n])

            objectives = objectives_[closest]
            variables = evolver.population.individuals[closest]

            ### DBSCAN
            # dbscan = DBSCAN(eps=0.125, min_samples=3).fit(objectives_)
            # core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
            # core_samples_mask[dbscan.core_sample_indices_] = True
            # labels_dbscan = dbscan.labels_
            # n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)

            # print("n clusters:", n_clusters_dbscan)
            # print(labels_dbscan)

            ### Gaussian mixtures
            # gmm = GaussianMixture(n_components=10, covariance_type="spherical").fit(objectives_)
            # labels_gmm = gmm.predict(objectives_)

            # print(labels_gmm)
            # print(gmm.means_)
            # print(gmm.covariances_)
            # print(gmm.converged_)

            # objectives = objectives_
            # variables = evolver.population.individuals

            color_data = color_solutions(
                objectives, ref_point=color_point, ideal=evolver.population.ideal_objective_vector,
            )
            obj_with_col_and_var = np.hstack(
                (objectives, color_data, variables, np.repeat(np.atleast_2d(n_iteration), len(objectives), axis=0))
            )
            # print(f"Computed rows: {obj_with_col_and_var}")

            # send response

            d["DATA"] = (
                np.array2string(
                    obj_with_col_and_var,
                    separator=",",
                    formatter={"all": lambda x: str(int(x)) if x.is_integer() else "{:.6f}".format(x)},
                )
                .replace("\n", "")
                .replace(" ", "")
            )

            for i in range(n_clusters):
                color_data = color_solutions(
                    labelled_objectives[i], ref_point=color_point, ideal=evolver.population.ideal_objective_vector,
                )
                obj_with_col_and_var = np.hstack(
                    (
                        labelled_objectives[i],
                        color_data,
                        labelled_variables[i],
                        np.repeat(np.atleast_2d(n_iteration), len(labelled_objectives[i]), axis=0),
                    )
                )

                d[f"{i}"] = (
                    np.array2string(
                        obj_with_col_and_var,
                        separator=",",
                        formatter={"all": lambda x: str(int(x)) if x.is_integer() else "{:.6f}".format(x)},
                    )
                    .replace("\n", "")
                    .replace(" ", "")
                )

            # send response
            data = parse_dict(d).encode("utf-8")

            # print(f"Data: {d['DATA']}")
            print(f"Sending data: {data}")

            connection.send(data)
            n_iteration += 1

    except Exception as e:
        # make sure to close the connection in case of errors
        print(e)
        connection.close()
