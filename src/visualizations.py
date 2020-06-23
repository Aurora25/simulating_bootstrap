import matplotlib.pyplot as plt


def plot_confidence_intervals_col(sim_res, pop_metric, metric):
    ax = plt.figure(figsize=[16, 8])
    for i, single_res in enumerate(sim_res):
        sim_metric_conf = single_res[metric].metric_confidence
        plt.plot([i, i], [sim_metric_conf.lower_bound, sim_metric_conf.upper_bound])
    plt.hlines(pop_metric[metric], xmin=0, xmax=len(sim_res), label="population_metric")
    plt.title(f"Confidence Intervals over population metric {metric}", size=20)
    ax.legend()


def plot_confidende_intervals_highlight(sim_res, pop_metric, metric):
    ax = plt.figure(figsize=[16, 8])
    in_label = False
    out_label = False

    label_lines = []
    labels = []
    for i, single_res in enumerate(sim_res):
        if single_res[metric].pop_metric_in_conf:
            line, = plt.plot([i, i], [single_res[metric].metric_confidence.lower_bound,
                                      single_res[metric].metric_confidence.upper_bound], color="b")
            if not in_label:
                in_label = True
                labels.append("in_confidence_interval")
                label_lines.append(line)
        else:
            line, = plt.plot([i, i], [single_res[metric].metric_confidence.lower_bound,
                                      single_res[metric].metric_confidence.upper_bound], color="r")
            if not out_label:
                out_label = True
                labels.append("out_confidence_interval")
                label_lines.append(line)
    pop_line = plt.hlines(pop_metric[metric], xmin=0, xmax=len(sim_res))

    label_lines.append(pop_line)
    labels.append("population_metric")

    ax.legend(tuple(label_lines), tuple(labels))
    plt.title(f"Confidence Intervals color-coded over population metric {metric}", size=20)
