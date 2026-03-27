# Neuro Data Format Guidelines

## Overview

This doc provides data schema explanations for C. elegans calcium imaging data processing. It ensures correct data structure usage.


## Core Data Schemas

### Format: Tidy DataFrame (`neuron_segments_df`)
**Use Case:** Statistical plotting, Seaborn/Plotly visualizations, GroupBy aggregations
**Structure:** Long-format Pandas DataFrame

**Required Columns:**
- `neuron`: neuron identifier (e.g., 'AWC', 'ASE')
- `stimulus`: stimulus identifier
- `time_point`: time index for signal values
- `delta_F_over_F0`: scalar calcium signal value
- `worm_key`: unique worm identifier
- `date`: experiment date
- `segment_index`: index of the segment in an experiment(neuron, worm, data)
- `stim_name`: stimulus name
- `stim_color`: stimulus color identifier
 

> Note: `worm_key` is just for the date, so the same `worm_key` in different dates represents different worm individuals. It is not a unique identifier across the entire dataset. However, using `worm_key`, `segment_index` and `date` together can uniquely identify a trial segment.

## Dive into the details:
- For `neuron` column, there're 11 pairs of neurons:
    - 'ADFL', 'ADFR', 'ADLL', 'ADLR', 'ASEL', 'ASER', 'ASGL', 'ASGR', 'ASHL', 'ASHR', 'ASIL', 'ASIR', 'ASJL', 'ASJR', 'ASKL', 'ASKR', 'AWAL', 'AWAR', 'AWBL', 'AWBR', 'AWCOFF', 'AWCON'.
- For `stimulus` column, it's actually the stimulus index, which I use snake_case to represent the stimulus type and order. For example, `b1_1` means 1 bacteria sample and its name is in `stim_name` column.
- If you look for neurons in the same trial, usually you can't get all neurons in the same trial, because of the technical issues during data collection. So you may find some neurons are missing in some trials. But different trials together can cover all neurons. 
- For `delta_F_over_F0` column, it's the scalar value of the calcium signal. I will talk about it together with `time_point` column. The time resolution is 1 second, and the time window is from 0 to 44, so there are 45 time points in total. 0-5 is the baseline period, 6-15 is the stimulus period, and 16-44 is the post-stimulus period.
- For `segment_index` column, it's the real experiment segment index in an individual recording.
- For `stim_name` column, it's the stimulus name, which is more human readable than `stimulus` column. For example, `b1_1` in `stimulus` column corresponds to `Bacteria 1` in `stim_name` column.
- For `stim_color` column, it's the stimulus color identifier, which is used for plotting. For example, `b1_1` in `stimulus` column corresponds to `#1f77b4` in `stim_color` column.
