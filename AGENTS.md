# Repository Agent Instructions

## 1. Purpose and Scope

This repository supports investigator-led scientific analysis. Work here is primarily analytical and exploratory, not the execution of a predefined workflow. These instructions apply throughout the repository.

## 2. Investigator Control

- The user determines the research questions, analysis order, methods, parameters, execution timing, interpretation, and when to proceed to another step.
- Address the user's current request without independently extending it into subsequent analyses, a project plan, or an end-to-end pipeline.
- Treat an explicit request to perform a broader analysis or advance to the next step as authorization for that specific work.

## 3. Agent Responsibilities

- Provide code or scripts for the requested analysis and diagnose or fix bugs identified by the user.
- Prefer small, readable, maintainable solutions. Avoid introducing frameworks, command-line interfaces, batch-processing systems, or dependencies unless the user requests them or they are necessary for the current task.
- When fixing a bug, identify its cause and its effect on the analysis, make the smallest appropriate change, and verify the affected behavior.

## 4. Notebook-First Analysis

- Jupyter Notebooks are the primary environment for analysis. Prefer code cells that fit into an existing Notebook or small functions that a Notebook can call.
- Create a new Notebook or substantially restructure an existing one only when requested.
- Make inputs, outputs, adjustable parameters, array dimensions, units, and material scientific assumptions clear in the code or accompanying explanation.

## 5. Execution and Data Integrity

- Do not run an entire Notebook, process a full dataset, generate final results, or write scientific conclusions unless the user requests that work. Focused checks with a small sample or minimal reproduction are permitted to validate a requested code change; state what was checked.
- Do not modify raw data. Keep raw data, processed data, and analysis outputs separate, and record processing parameters that affect results.
- Ask a concise clarifying question when an unknown would materially change the method or scientific interpretation. Otherwise, state the assumption and complete the current request.

## 6. Scientific Figures

- Prioritize scientific accuracy over visual appeal. Represent the data, uncertainty, and material limitations faithfully.
- Use English for all figure text, including titles, axis labels, legends, annotations, and captions, unless the user explicitly requests another language.
- Give each presentation figure one clear scientific message. Show the result and the comparison needed to understand it; do not reproduce the analysis workflow by default.
- Make the figure understandable primarily through the plotted data, layout, and visual emphasis. Readers should be able to identify the main comparison without reading explanatory paragraphs.
- Avoid in-plot explanatory annotations by default. If an annotation is essential to interpret the evidence correctly, keep it brief. Do not use text to compensate for an unclear plot or overcrowded layout.
- Keep titles, labels, and legends short. Retain necessary axis labels, units, and keys; place methods, selection rules, and detailed limitations in a concise external caption or accompanying text.
- Separate presentation figures from data-inspection resources. Full matrices, exhaustive neuron or condition panels, and multi-page atlases belong in supporting resources unless completeness is the purpose of the requested figure. Disclose any example-selection rule in the caption.
- Introduce model comparisons through the scientific question they test. Do not use unexplained model identifiers as the main narrative or assume that readers know the fitting workflow.
- Remove visual elements and repeated labels that do not help readers understand the result. If the message remains scattered across too many panels, simplify the figure's scope or structure before adding annotations.
- Use a consistent visual language across related figures, including colors, symbols, typography, and terminology. Use comparable scales where direct comparisons require them.
- Minimize text without hiding uncertainty, missing data, coverage limits, or differences in scale. The figure and its concise caption must preserve the boundaries of the evidence.
