import json
import math
import random
from datetime import datetime, timedelta

import numpy as np
from sortedcontainers import SortedDict

from utils import read_json

from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors


def similarity(x, y, index):
    sum_xx, sum_xy, sum_yy = 0, 0, 0
    for i in index:
        xi = x[i] if i in x else 0
        yi = y[i] if i in y else 0

        sum_xx += xi * xi
        sum_xy += xi * yi
        sum_yy += yi * yi

    return sum_xy / (math.sqrt(sum_xx * sum_yy) + 1e-6)


def extract_sample(x, index):
    v = []
    for i in index:
        if i in x:
            v.append(x[i])
        else:
            v.append(0)

    return np.array(v)


def add_to_center(center, lower_date, upper_date, data, positive=True):
    total = 0
    for d in data.irange(lower_date, upper_date, (True, False)):
        total += data[d]

        if d in center:
            if positive:
                center[d] += data[d]
            else:
                center[d] -= data[d]
        else:
            center[d] = data[d]

    return total


def adjust_cluster(min_date, current_date, next_date, data, last_assignment, next_cluster, centers,
                   cluster_totals, total_queries, cluster_sizes, rho):
    n = (next_date - min_date).seconds // 60 + (next_date - min_date).days * 1440 + 1
    print(n)
    num_sample = 10000
    if n > num_sample:
        index = random.sample(range(0, n), num_sample)
    else:
        index = range(0, n)

    # per min distribution
    index = [min_date + timedelta(minutes=i) for i in index]

    new_assignment = last_assignment.copy()
    for cluster in centers.keys():
        for template in last_assignment:
            if last_assignment[template] == cluster:
                cluster_totals[cluster] += add_to_center(centers[cluster], current_date, next_date, data[template])

    print("Building kdtree for single point assignment")
    clusters = sorted(centers.keys())

    samples = []

    for cluster in clusters:
        sample = extract_sample(centers[cluster], index)
        samples.append(sample)

    if len(samples) == 0:
        nearest_neighbors = None
    else:
        normalized_samples = normalize(np.array(samples), copy=False)
        nearest_neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", metric='l2')
        nearest_neighbors.fit(normalized_samples)
    print("Finish building kdtree for single point assignment")

    count = 0
    for template in sorted(data.keys()):
        count += 1
        # Test whether this template still belongs to the original cluster
        if new_assignment[template] != -1:
            center = centers[new_assignment[template]]
            # print(cnt, new_ass[t], Similarity(data[t], center, index))
            if cluster_sizes[new_assignment[template]] == 1 or similarity(data[template], center, index) > rho:
                continue

        # the template is eliminated from the original cluster
        if new_assignment[template] != -1:
            cluster = new_assignment[template]
            cluster_sizes[cluster] -= 1
            add_to_center(centers[cluster], min_date, next_date, data[template], False)
            print("%s: template %s quit from cluster %d with total %d" % (next_date, count, cluster,
                                                                          total_queries[template]))

        # Whether this template has "arrived" yet?
        if new_assignment[template] == -1 and len(list(data[template].irange(current_date, next_date))) == 0:
            continue

        # whether this template is similar to the center of an existing cluster
        new_cluster = None

        if nearest_neighbors is None:
            for cluster in centers.keys():
                center = centers[cluster]
                if similarity(data[template], center, index) > rho:
                    new_cluster = cluster
                    break
        else:
            nbr = nearest_neighbors.kneighbors(
                normalize([extract_sample(data[template], index)]), return_distance=False)[0][0]
            if similarity(data[template], centers[clusters[nbr]], index) > rho:
                new_cluster = clusters[nbr]

        if new_cluster is not None:
            if new_assignment[template] == -1:
                print("%s: template %s joined cluster %d with total %d" % (next_date, count,
                                                                           new_cluster, total_queries[template]))
            else:
                print("%s: template %s reassigned to cluster %d with total %d" % (next_date,
                                                                                  count, new_cluster,
                                                                                  total_queries[template]))
            new_assignment[template] = new_cluster
            add_to_center(centers[new_cluster], min_date, next_date, data[template])
            cluster_sizes[new_cluster] += 1
            continue

        if new_assignment[template] == -1:
            print("%s: template %s created cluster as %d with total %d" % (next_date, count,
                                                                           next_cluster, total_queries[template]))
        else:
            print("%s: template %s recreated cluster as %d with total %d" % (next_date, count,
                                                                             next_cluster, total_queries[template]))

        new_assignment[template] = next_cluster
        centers[next_cluster] = SortedDict()
        add_to_center(centers[next_cluster], min_date, next_date, data[template])
        cluster_sizes[next_cluster] = 1
        cluster_totals[next_cluster] = 0
        next_cluster += 1

    clusters = list(centers.keys())
    # a union-find set to track the root cluster for clusters that have been merged
    root = [-1] * len(clusters)

    print("Building kdtree for cluster merging")

    samples = list()

    for cluster in clusters:
        sample = extract_sample(centers[cluster], index)
        samples.append(sample)

    if len(samples) == 0:
        nearest_neighbors = None
    else:
        normalized_samples = normalize(np.array(samples), copy=False)
        nearest_neighbors = NearestNeighbors(n_neighbors=2, algorithm="kd_tree", metric='l2')
        nearest_neighbors.fit(normalized_samples)

    print("Finish building kdtree for cluster merging")

    for i in range(len(clusters)):
        c1 = clusters[i]
        c = None

        if nearest_neighbors is None:
            for j in range(i + 1, len(clusters)):
                c2 = clusters[j]
                if similarity(centers[c1], centers[c2], index) > rho:
                    c = c2
                    break
        else:
            nbr = nearest_neighbors.kneighbors([extract_sample(centers[c1], index)], return_distance=False)[0]

            if clusters[nbr[0]] == c1:
                nbr = nbr[1]
            else:
                nbr = nbr[0]

            while root[nbr] != -1:
                nbr = root[nbr]

            if c1 != clusters[nbr] and similarity(centers[c1], centers[clusters[nbr]], index) > rho:
                c = clusters[nbr]

        if c is not None:
            add_to_center(centers[c], min_date, next_date, centers[c1])
            cluster_sizes[c] += cluster_sizes[c1]

            del centers[c1]
            del cluster_sizes[c1]

            if nearest_neighbors is not None:
                root[i] = nbr

            for template in data.keys():
                if new_assignment[template] == c1:
                    new_assignment[template] = c
                    print("%d assigned to %d with total %d" % (c1, c, total_queries[template]))

            print("%s: cluster %d merged into cluster %d" % (next_date, c1, c))

    return new_assignment, next_cluster


def online_clustering(min_date, max_date, data, total_queries, rho):
    cluster_gap = 1440
    n = (max_date - min_date).seconds // 60 + (max_date - min_date).days * 1440 + 1
    num_gaps = n // cluster_gap
    print(n)
    print(num_gaps)

    centers = {}
    cluster_totals = {}
    cluster_sizes = {}

    assignments = []
    min_date_assignment = {}
    for template in data:
        min_date_assignment[template] = -1
    assignments.append((min_date, min_date_assignment))

    current_date = min_date
    next_cluster = 0
    for i in range(num_gaps):
        next_date = current_date + timedelta(minutes=cluster_gap)
        # Calculate similarities based on arrival rates up to the past month
        month_min_date = max(min_date, next_date - timedelta(days=30))
        assign, next_cluster = adjust_cluster(month_min_date, current_date, next_date, data, assignments[-1][1],
                                              next_cluster, centers, cluster_totals, total_queries, cluster_sizes, rho)
        assignments.append((next_date, assign))
        current_date = next_date

    return next_cluster, assignments, cluster_totals


# The number of the largest clusters to consider for coverage evaluation and forecasting
MAX_CLUSTER_NUM = 5

# If it's the full trace used for kernel regression, always aggregate the data
# into 10 minutes intervals
FULL = True

if FULL:
    AGGREGATE = 10
else:
    AGGREGATE = 1

# If it's the noisy data evaluation, use a smaller time gap to calculate the
# total volume of the largest clusters. In the future we should automatically
# adjust this to the point where the worklaod has shifted after we detect that a
# shift happened (i.e., the majority of the workload comes from unseen queries).
# And of course a long horizon prediction is hard to work if the shift only
# happened for a short period.
NOISE = False

if NOISE:
    LAST_TOTAL_TIME_GAP = 1200  # seconds
else:
    LAST_TOTAL_TIME_GAP = 86400  # seconds


def generate_cluster_coverage(min_date, max_date, data, data_aggr, templates, assignments, total_queries, num_clusters):
    coverage_lists = [[] for i in range(MAX_CLUSTER_NUM)]
    top_clusters = []
    online_clusters = dict()
    last_date = min_date
    if FULL:
        # Normal full evaluation
        assignments = assignments[0:]

    for current_date, cluster_assignments in assignments:
        cluster_totals = dict()
        date_total = 0
        for template, cluster in cluster_assignments.items():
            if cluster == -1:
                continue

            last_total_date = next(data_aggr[template].irange(maximum=current_date, reverse=True))
            if (current_date - last_total_date).seconds < LAST_TOTAL_TIME_GAP:
                template_total = data_aggr[template][last_total_date]
            else:
                template_total = 0
            date_total += template_total

            if cluster in cluster_totals:
                cluster_totals[cluster] += template_total
            else:
                cluster_totals[cluster] = template_total

        if len(cluster_totals) == 0:
            last_date = current_date
            continue

        sorted_clusters = sorted(cluster_totals.items(), key=lambda x: x[1], reverse=True)

        sorted_names, sorted_totals = zip(*sorted_clusters)

        current_top_clusters = sorted_clusters[:MAX_CLUSTER_NUM]
        print(current_date, current_top_clusters)

        if FULL:
            record_ahead_time = timedelta(days=30)
        else:
            record_ahead_time = timedelta(days=8)

        for c, v in current_top_clusters:
            if c in online_clusters:
                continue
            online_clusters[c] = SortedDict()
            for template, cluster in cluster_assignments.items():
                if cluster != c:
                    continue
                if FULL:
                    start_date = min_date
                else:
                    start_date = max(min_date, last_date - timedelta(weeks=4))
                for d in data[template].irange(start_date, last_date + record_ahead_time, (True, False)):
                    if d in online_clusters[cluster]:
                        online_clusters[cluster][d] += data[template][d]
                    else:
                        online_clusters[cluster][d] = data[template][d]

        current_top_cluster_names = next(zip(*current_top_clusters))
        for template, cluster in cluster_assignments.items():
            if not (cluster in current_top_cluster_names):
                continue

            for d in data[template].irange(last_date + record_ahead_time, current_date +
                                                                          record_ahead_time, (True, False)):
                if d in online_clusters[cluster]:
                    online_clusters[cluster][d] += data[template][d]
                else:
                    online_clusters[cluster][d] = data[template][d]

        top_clusters.append((current_date.isoformat(), current_top_clusters))

        for i in range(MAX_CLUSTER_NUM):
            coverage_lists[i].append(sum(sorted_totals[:i + 1]) / date_total)

        last_date = current_date

    coverage = [sum(l) / len(l) for l in coverage_lists]

    result = {}
    for c in online_clusters:
        if len(online_clusters[c]) < 2:
            continue
        l = online_clusters[c].keys()[0]
        r = online_clusters[c].keys()[-1]

        n = (r - l).seconds // 60 + (r - l).days * 1440 + 1
        dates = [l + timedelta(minutes=i) for i in range(n)]
        v = 0
        # for d, v in online_clusters[c].items():
        result[c] = []
        for d in dates:
            if d in online_clusters[c]:
                v += online_clusters[c][d]
            if d.minute % AGGREGATE == 0:
                result[c].append((d.isoformat(), v))
                v = 0

    return result, top_clusters, coverage


if __name__ == '__main__':
    templates = read_json("templated_queries.json")

    template_queries = []
    total_queries = {}
    min_date = datetime.max
    max_date = datetime.min
    data = {}
    data_aggr = dict()

    for template in templates:
        template_timestamps = templates[template]
        num_queries = sum(template_timestamps.values())
        total_queries[template] = num_queries
        template_queries.append(template)
        data[template] = SortedDict()
        data_aggr[template] = SortedDict()

        total = 0

        for timestamp in template_timestamps:
            datetime_timestamp = datetime.fromisoformat(timestamp)
            count = template_timestamps[timestamp]

            data[template][datetime_timestamp] = count

            total += count
            data_aggr[template][datetime_timestamp] = total

            min_date = min(min_date, datetime_timestamp)
            max_date = max(max_date, datetime_timestamp)

    template_queries = sorted(template_queries)

    # The threshold to determine whether a query template belongs to a cluster.
    rho = 0.8
    next_cluster, assignments, cluster_totals = online_clustering(min_date, max_date, data, total_queries, rho)
    # print(next_cluster)
    # print(assignments)
    # print(cluster_totals)

    result, top_clusters, coverage = generate_cluster_coverage(min_date, max_date, data, data_aggr, templates,
                                                               assignments, total_queries, next_cluster)
    # print(result)
    print(top_clusters)
    print(coverage)

    with open("result_clusters.json", "w") as f:
        f.write(json.dumps(result))

    with open("top_clusters.json", "w") as f:
        f.write(json.dumps(top_clusters))
