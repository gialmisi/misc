import socket
import sys

import numpy as np
import pandas as pd
from desdeo_emo.EAs.RVEA import RVEA
from desdeo_problem.Objective import _ScalarObjective
from desdeo_problem.Problem import MOProblem
from desdeo_problem.Variable import variable_builder
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from myparser import parse_dict, parse_message
from solution_color import color_solutions

n_clusters = 10

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
            # fit to n_clusters and find the closest solutions to each cluster's centroid
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(objectives_)
            closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, objectives_)
            objectives = objectives_[closest]
            variables = evolver.population.individuals[closest]

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
            # d["DATA"] = np.array2string(obj_with_col_and_var, separator=",").replace("\n", "").replace(" ", "")
            # d["DATA"] = np.array2string(objectives, separator=",").replace("\n", "").replace(" ", "")
            print(d["DATA"])

            data = parse_dict(d).encode("utf-8")

            print(f"Sending data: {data}")

            connection.send(data)
            n_iteration += 1

    except Exception as e:
        # make sure to close the connection in case of errors
        print(e)
        connection.close()
