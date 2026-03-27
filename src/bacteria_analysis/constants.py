"""Shared constants for preprocessing and test fixtures."""

REQUIRED_COLUMNS = (
    "neuron",
    "stimulus",
    "time_point",
    "delta_F_over_F0",
    "worm_key",
    "segment_index",
    "date",
    "stim_name",
    "stim_color",
)

EXPECTED_TIMEPOINTS = tuple(range(45))
BASELINE_TIMEPOINTS = (0, 1, 2, 3, 4, 5)
NEURON_ORDER = (
    "ADFL",
    "ADFR",
    "ADLL",
    "ADLR",
    "ASEL",
    "ASER",
    "ASGL",
    "ASGR",
    "ASHL",
    "ASHR",
    "ASIL",
    "ASIR",
    "ASJL",
    "ASJR",
    "ASKL",
    "ASKR",
    "AWAL",
    "AWAR",
    "AWBL",
    "AWBR",
    "AWCOFF",
    "AWCON",
)

