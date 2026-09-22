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
