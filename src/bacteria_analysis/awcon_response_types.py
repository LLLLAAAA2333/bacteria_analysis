"""Exploratory neural temporal phenotypes, defined without chemical information."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TYPES = ("stimulus", "post", "sustained", "biphasic", "unresolved")
COLORS = dict(zip(TYPES, ["#D98632", "#437FA6", "#A54363", "#6B59A5", "#AAAAAA"]))
KEYS = ["sample_id", "date", "worm_key"]
BILATERAL_CLASSES = ("ASK", "ADL", "ASI", "AWA", "AWB", "ASG", "ADF", "ASH", "ASJ")


def load_trials(path, neuron="AWCON"):
    """Rows = strain/date/animal/trial; columns = 45 complete volumes, -5..39 s.

    As in cell47, bilateral classes average available L/R values within each
    trial/volume. This estimates a class response and does not test lateralization.
    Exact identities such as ASHL, ASEL and AWCON remain separate.
    """
    source_neurons = [neuron + side for side in ("L", "R")] if neuron in BILATERAL_CLASSES else [neuron]
    data = pd.read_parquet(
        path, columns=["date", "worm_key", "segment_index", "stim_name", "neuron",
                       "time_point", "delta_F_over_F0", "start_time", "end_time"],
        filters=[("neuron", "in", source_neurons), ("time_point", ">=", 0), ("time_point", "<", 45)],
    )
    if data.empty:
        raise ValueError(f"No records for {neuron}; searched raw neuron identities {source_neurons}")
    if not data.start_time.eq(5).all() or not data.end_time.eq(15).all():
        raise ValueError(f"{neuron}: expected onset=5 and exclusive offset=15")
    data["sample_id"] = data.stim_name.astype(str).str.extract(r"^(A\d{3})", expand=False)
    keys = KEYS + ["segment_index"]
    if data[keys + ["time_point", "neuron"]].isna().any().any():
        raise ValueError("Missing trial/neuron identity or invalid Axxx strain identity")
    for column in KEYS:
        data[column] = data[column].astype(str)
    if data.duplicated(keys + ["neuron", "time_point"]).any():
        raise ValueError("Duplicate neuron/volume records within a trial")
    data["delta_F_over_F0"] = data.delta_F_over_F0.replace([np.inf, -np.inf], np.nan)
    grouped = data.groupby(keys + ["time_point"], observed=True).delta_F_over_F0
    trials = grouped.mean().unstack("time_point").reindex(columns=range(45)).sort_index()
    counts = grouped.count().unstack("time_point").reindex(index=trials.index, columns=range(45)).fillna(0)
    trials.attrs["neural_input"] = {
        "requested_neuron": neuron, "requested_raw_identities": source_neurons,
        "observed_raw_identities": sorted(data.neuron.unique().tolist()),
        "pooling": "available-side mean within trial/volume; sides are not independent replicates",
        "raw_rows": len(data), "n_trials": len(trials),
        "trial_volume_counts_by_finite_neurons": {
            str(int(n)): int(total) for n, total in pd.Series(counts.to_numpy().ravel()).value_counts().sort_index().items()
        },
        "trials_with_all_requested_neurons_at_every_volume": int(counts.eq(len(source_neurons)).all(axis=1).sum()),
    }
    return trials


def template_library():
    """Six fixed variants per family, and both polarities; no fitting to this panel.

    Each vector contains means in [0,5), ... [35,40) s. These are descriptive
    shapes, not mechanistic response models or estimated calcium kernels.
    """
    patterns = {
        "stimulus": [[1,.2,0,0,0,0,0,0], [1,.7,.1,0,0,0,0,0], [.4,1,.15,0,0,0,0,0],
                     [1,1,.15,.05,0,0,0,0], [.1,1,.05,0,0,0,0,0], [.8,1,.3,.05,0,0,0,0]],
        "post": [[0,0,1,.2,0,0,0,0], [0,0,1,.7,.3,.1,0,0], [0,0,.3,1,.6,.2,0,0],
                 [0,0,0,.2,1,.7,.3,.1], [0,0,.2,.5,1,1,.8,.5], [0,0,0,0,.2,.5,1,.7]],
        "sustained": [[.5,1,.8,.2,0,0,0,0], [.2,.8,1,.5,.2,0,0,0], [.6,1,1,.7,.4,.2,.1,0],
                      [.3,.8,1,1,.8,.6,.4,.2], [1,1,1,1,.9,.7,.5,.3], [.7,1,.6,.3,.1,0,0,0]],
        "biphasic": [[.5,1,0,-.7,-1,-.5,-.2,0], [.2,1,.5,-.2,-.5,-.3,-.1,0],
                     [.2,.5,0,-.5,-1,-1,-.8,-.5], [.2,.8,1,.4,-.3,-.7,-1,-.8],
                     [0,.5,1,.5,-.1,-.4,-.5,-.4], [1,.5,0,-.5,-.2,0,0,0]],
    }
    rows = []
    for kind, variants in patterns.items():
        for sign in (1, -1):
            direction = ("up_then_down" if sign > 0 else "down_then_up") if kind == "biphasic" else ("up" if sign > 0 else "down")
            for i, vector in enumerate(variants):
                rows.append(dict(template_id=f"{kind}:{direction}:{i + 1}", response_type=kind, direction=direction,
                                 **{f"bin_{j * 5}": value * sign for j, value in enumerate(vector)}))
    return pd.DataFrame(rows)


def classify_bins(values, noise_sd, settings):
    """Fit y = a * template, a >= 0; keep a provisional shape even for weak signals.

    Fits preserve zero and stimulus alignment. Score = explained uncentered
    energy, not a calibrated probability. Noise, fit quality and ambiguity are
    separate flags, and never change a non-flat trace into 'no response'.
    """
    values = np.asarray(values, dtype=float)
    out = dict(response_type="unresolved", direction="uncertain", reason="", quality_flags="",
               peak_bin_amplitude=np.nan, detect_threshold=np.nan, signal_clear=False,
               shape_fit=np.nan, shape_margin=np.nan, template_id="", second_type="", second_direction="")
    if len(values) != 8 or not np.isfinite(values).all():
        out["reason"] = "incomplete trace"
        return out
    energy = float(values @ values)
    if energy < 1e-20:
        out["reason"] = "numerically flat trace"
        return out
    library = template_library()
    basis = library[[f"bin_{j}" for j in range(0, 40, 5)]].to_numpy()
    products = np.maximum(basis @ values, 0)
    scores = products**2 / ((basis**2).sum(axis=1) * energy)
    library["score"] = np.clip(scores, 0, 1)
    # Compare family/direction maxima, so near-identical variants within a family
    # do not create artificial ambiguity in the runner-up score.
    best = library.sort_values("score", ascending=False, kind="stable").drop_duplicates(["response_type", "direction"])
    first, second = best.iloc[0], best.iloc[1]
    peak = float(np.max(np.abs(values)))
    detect = max(settings["noise_multiplier"] * noise_sd, settings["absolute_detection_floor"]) if np.isfinite(noise_sd) else np.nan
    clear = bool(np.isfinite(detect) and peak >= detect)
    flags = []
    if not clear:
        flags.append("low_signal_or_missing_baseline")
    if first.score < settings["min_shape_fit"]:
        flags.append("poor_template_fit")
    if first.score - second.score < settings["min_shape_margin"]:
        flags.append("ambiguous_shape")
    out.update(response_type=first.response_type, direction=first.direction, template_id=first.template_id,
               peak_bin_amplitude=peak, detect_threshold=detect, signal_clear=clear,
               shape_fit=float(first.score), shape_margin=float(first.score - second.score),
               second_type=second.response_type, second_direction=second.direction, quality_flags="; ".join(flags))
    return out


def consensus(rows, settings, equal_dates=False):
    """Keep the leading provisional phenotype; report heterogeneity separately."""
    counts = rows.groupby(["response_type", "direction"], dropna=False).size()
    if equal_dates:
        fractions = rows.groupby(["date", "response_type", "direction"]).size().div(rows.groupby("date").size(), level="date")
        votes = fractions.groupby(level=["response_type", "direction"]).sum() / rows.date.nunique()
    else:
        votes = counts / len(rows)
    eligible = votes.loc[votes.index.get_level_values("response_type") != "unresolved"]
    out = dict(response_type="unresolved", direction="uncertain", agreement=0.0,
               n_animals=len(rows), n_dates=rows.date.nunique(), n_resolved_animals=int(rows.response_type.ne("unresolved").sum()),
               leading_type="unresolved", leading_direction="uncertain", reason="no finite non-flat animal shape",
               signal_fraction=float(rows.signal_clear.mean()), median_shape_fit=float(rows.shape_fit.median()),
               median_shape_margin=float(rows.shape_margin.median()), secondary_phenotype="", secondary_fraction=0.0,
               quality_flags="", confidence="unresolved")
    if not eligible.empty:
        # A tie is retained as a quality flag; mean match score breaks exact ties
        # deterministically, without averaging opposite signed curves together.
        candidates = eligible.loc[np.isclose(eligible, eligible.max())].index
        quality = rows.groupby(["response_type", "direction"]).shape_fit.mean()
        winner = quality.reindex(candidates).sort_values(ascending=False, kind="stable").index[0]
        out.update(agreement=float(eligible.loc[winner]), leading_type=winner[0], leading_direction=winner[1],
                   response_type=winner[0], direction=winner[1], reason="")
        other = eligible.drop(winner)
        if not other.empty:
            out.update(secondary_phenotype=":".join(other.idxmax()), secondary_fraction=float(other.max()))
        flags = []
        if len(candidates) > 1:
            flags.append("tied_leading_patterns")
        if len(rows) < settings["min_animals"]:
            flags.append("limited_repeats")
        if out["agreement"] < settings["consensus_fraction"]:
            flags.append("mixed_animal_patterns")
        if out["signal_fraction"] < settings["min_signal_fraction"]:
            flags.append("low_signal_coverage")
        if out["median_shape_fit"] < settings["min_shape_fit"]:
            flags.append("poor_template_fit")
        if out["median_shape_margin"] < settings["min_shape_margin"]:
            flags.append("ambiguous_shape")
        out.update(quality_flags="; ".join(flags), confidence="supported" if not flags else "provisional")
    out["phenotype"] = out["response_type"] + ":" + out["direction"]
    return out


def type_responses(trials, settings):
    """Trial means -> animal curves; retain trial baseline noise without sqrt(n) scaling."""
    if not (settings["noise_multiplier"] > 0 and settings["absolute_detection_floor"] > 0
            and 0 <= settings["min_shape_fit"] <= 1 and 0 <= settings["min_shape_margin"] <= 1
            and settings["min_animals"] >= 2 and .5 < settings["consensus_fraction"] <= 1
            and 0 <= settings["min_signal_fraction"] <= 1):
        raise ValueError("Invalid positive detection / fit / consensus quality settings")
    animals = trials.groupby(level=KEYS).mean()
    # Only five prestimulus volumes are available and F0 was fitted upstream.
    # Median trial baseline SD is a descriptive noise reference, not a null model.
    baseline_sd = trials.loc[:, 0:4].std(axis=1).where(trials.loc[:, 0:4].count(axis=1).eq(5))
    noise = baseline_sd.groupby(level=KEYS).median()
    counts = trials.groupby(level=KEYS).size()
    bins = pd.DataFrame({start: animals.loc[:, start + 5:start + 9].mean(axis=1)
                         for start in range(0, 40, 5)})
    full_coverage = animals.notna().all(axis=1)
    rows = []
    for key, curve in bins.iterrows():
        values = curve.to_numpy() if full_coverage.loc[key] else np.full(8, np.nan)
        result = classify_bins(values, noise.loc[key], settings)
        rows.append(dict(zip(KEYS, key), **result, baseline_sd=noise.loc[key], n_trials=int(counts.loc[key]),
                         baseline_mean=float(animals.loc[key, 0:4].mean()),
                         volume_coverage=float(animals.loc[key].notna().mean())))
    labels = pd.DataFrame(rows)
    date_rows = [dict(sample_id=sample, date=date, **consensus(group, settings))
                 for (sample, date), group in labels.groupby(["sample_id", "date"], sort=True)]
    strain_rows = [dict(sample_id=sample, **consensus(group, settings, equal_dates=True))
                   for sample, group in labels.groupby("sample_id", sort=True)]
    return animals, bins, labels, pd.DataFrame(date_rows), pd.DataFrame(strain_rows)


def save_type_plots(animals, animal_labels, strain_labels, folder, review_samples=(), neuron="AWCON"):
    """Raw units and zero-preserving normalized shapes remain visibly separate."""
    folder = Path(folder)
    labels = animal_labels.set_index(KEYS)
    raw = animals.loc[:, 5:44].copy()
    raw.columns = np.arange(40)
    normalized = raw.div(labels.peak_bin_amplitude.where(labels.response_type.ne("unresolved")), axis=0)
    shape_means = normalized.groupby(level=["sample_id", "date"]).mean().groupby(level="sample_id").mean()
    raw_means = raw.groupby(level=["sample_id", "date"]).mean().groupby(level="sample_id").mean()
    table = strain_labels.copy()
    table["type_order"] = table.response_type.map({name: i for i, name in enumerate(TYPES)})
    table = table.sort_values(["type_order", "direction", "agreement", "sample_id"], ascending=[True, True, False, True])
    ids = table.sample_id.tolist()
    shape_means.loc[table.loc[table.response_type.eq("unresolved"), "sample_id"]] = np.nan
    fig, axes = plt.subplots(1, 4, figsize=(17, max(7, len(ids) * .14)),
                             gridspec_kw={"width_ratios": [5, 5, 2.0, 1.2]}, sharey=True, layout="constrained")
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#DDDDDD")
    for ax, values, limit, title in zip(axes[:2], [raw_means, shape_means], [1.5, 1.0],
                                      ["Original strain means (color saturates)", "Equal-animal normalized shapes (including weak)"]):
        im = ax.imshow(values.reindex(ids), aspect="auto", cmap=cmap, vmin=-limit, vmax=limit,
                       extent=(0, 40, len(ids) - .5, -.5), interpolation="nearest")
        ax.axvline(10, color="#444444", ls="--", lw=.8)
        ax.set(title=title, xlabel="Time from stimulus onset (s)")
        fig.colorbar(im, ax=ax, shrink=.3, label="Delta F/F0" if ax is axes[0] else "Shape (positive scale division)", extend="both")
    axes[0].set_yticks(range(len(ids)), [f"{r.sample_id}{'*' if r.confidence != 'supported' else ''}  {r.response_type}/{r.direction}" for r in table.itertuples()], fontsize=6)
    fractions = pd.crosstab([animal_labels.sample_id, animal_labels.date], animal_labels.response_type, normalize="index")
    fractions = fractions.groupby(level="sample_id").mean().reindex(index=ids, columns=TYPES[:-1], fill_value=0)
    axes[2].imshow(fractions.fillna(0), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axes[2].set(xticks=range(4), xticklabels=TYPES[:-1], title="Individual type\nfractions (0–1)")
    axes[2].tick_params(axis="x", rotation=90, labelsize=7)
    axes[3].barh(range(len(ids)), table.agreement, color=table.response_type.map(COLORS))
    axes[3].set(xlim=(0, 1), xlabel="Agreement", title="Leading pattern\nagreement")
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False, labelleft=False)
    fig.suptitle(f"{neuron} provisional dynamic types | * quality/repetition flags | one volume = 1 s", fontsize=11)
    fig.savefig(folder / "type_atlas.png", dpi=150)
    plt.show()

    present = [kind for kind in TYPES[:-1] if table.response_type.eq(kind).any()]
    examples = []
    for kind in present:
        subset = table.loc[table.response_type.eq(kind)].sort_values(
            ["agreement", "n_animals", "sample_id"], ascending=[False, False, True])
        examples.append(subset.iloc[0].sample_id)
    examples += [sample for sample in review_samples if sample in set(table.sample_id) and sample not in examples]
    if examples:
        fig, axes = plt.subplots(len(examples), 2, figsize=(12, 2.8 * len(examples)), squeeze=False, layout="constrained")
        for i, sample in enumerate(examples):
            row = table.set_index("sample_id").loc[sample]
            for ax, values, title in zip(axes[i], [raw, normalized], ["Original animal curves", "Normalized animal curves"]):
                curves = values.xs(sample, level="sample_id")
                individual_types = labels.xs(sample, level="sample_id").response_type
                for key, curve in curves.iterrows():
                    ax.plot(range(40), curve, color=COLORS[individual_types.loc[key]], alpha=.45, lw=.8)
                avg = curves.groupby(level="date").mean().mean(axis=0)
                ax.plot(range(40), avg, color="black", lw=1.5)
                ax.axvspan(0, 10, color="#AAAAAA", alpha=.12)
                ax.axhline(0, color="#999999", lw=.5)
                ax.set(title=f"{sample}: {row.response_type}/{row.direction}\n{title}; agreement {row.agreement:.0%}", xlabel="Time (s)",
                       ylabel="Delta F/F0" if title.startswith("Original") else "Normalized shape")
                ax.title.set_fontsize(10)
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([0], [0], color=COLORS[kind], label=kind) for kind in TYPES],
                   loc="outside lower center", ncol=5, fontsize=8)
        fig.suptitle("Representatives + fixed review strains; thin: animals colored by their type; black: date-equal mean", fontsize=10)
        fig.savefig(folder / "type_examples.png", dpi=150)
        plt.show()
    return examples


def plot_support(date_labels, strain_labels, sensitivity, folder):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    counts = strain_labels.response_type.value_counts().reindex(TYPES, fill_value=0)
    axes[0].bar(TYPES, counts, color=[COLORS[name] for name in TYPES])
    for i, count in enumerate(counts):
        axes[0].text(i, count + .5, str(count), ha="center", fontsize=9)
    axes[0].set(title="Leading provisional type (all strains)", ylabel="Number of strains")
    axes[0].tick_params(axis="x", rotation=35)
    table = pd.crosstab(date_labels.response_type, date_labels.date).reindex(TYPES, fill_value=0)
    axes[1].imshow(table, cmap="Blues", aspect="auto")
    for i in range(len(table)):
        for j in range(len(table.columns)):
            axes[1].text(j, i, str(table.iloc[i, j]), ha="center", va="center", fontsize=8,
                         color="white" if table.iloc[i, j] > table.to_numpy().max() / 2 else "black")
    axes[1].set(title="Strain/date labels (no cross-date pooling)",
                xticks=range(len(table.columns)), xticklabels=table.columns,
                yticks=range(len(table)), yticklabels=table.index)
    axes[1].tick_params(axis="x", rotation=90)
    counts = pd.crosstab(sensitivity.scenario, sensitivity.confidence).reindex(columns=["supported", "provisional", "unresolved"], fill_value=0)
    counts = counts.reindex(sensitivity.scenario.drop_duplicates())
    bottom = np.zeros(len(counts))
    for kind, color in [("supported", "#438777"), ("provisional", "#BBBBBB"), ("unresolved", "#DDDDDD")]:
        axes[2].bar(range(len(counts)), counts[kind], bottom=bottom, label=kind, color=color)
        bottom += counts[kind].to_numpy()
    axes[2].set(title="Quality-flag sensitivity; shape labels unchanged", ylabel="Number of strains",
                xticks=range(len(counts)), xticklabels=counts.index)
    axes[2].tick_params(axis="x", rotation=60, labelsize=8)
    axes[2].legend(fontsize=7, loc="upper left")
    fig.savefig(Path(folder) / "type_support.png", dpi=150)
    plt.show()


def plot_templates(folder):
    library = template_library()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), layout="constrained")
    columns = [f"bin_{j}" for j in range(0, 40, 5)]
    for ax, kind in zip(axes, TYPES[:-1]):
        subset = library.loc[library.response_type.eq(kind) & library.direction.isin(["up", "up_then_down"])]
        for values in subset[columns].to_numpy():
            ax.plot(np.arange(2.5, 40, 5), values, color=COLORS[kind], alpha=.5, lw=1)
        ax.axvline(10, color="black", ls="--", lw=.6)
        ax.axhline(0, color="grey", lw=.5)
        ax.set(title=kind, xlabel="Time from onset (s)", ylim=(-1.1, 1.1))
    fig.suptitle("Fixed shape references: six variants/family, mirrored for opposite polarity; positive amplitude fitted freely", fontsize=10)
    fig.savefig(Path(folder) / "shape_templates.png", dpi=150)
    plt.show()
