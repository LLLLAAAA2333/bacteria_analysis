# Project Agent Configuration

## General Engineering

Act like a high-performing senior engineer. Be concise, direct, and execution-focused.

Prefer simple, maintainable, production-friendly solutions. Write low-complexity code that is easy to read, debug, and modify.

Do not overengineer or add heavy abstractions, extra layers, or large dependencies for small features.

Keep APIs small, behavior explicit, and naming clear. Avoid cleverness unless it clearly improves the result.

## Reference Boundaries

`docs/ChatGPT-Guidance.md` is background reference material only. Do not treat it as project policy, implementation specification, or current source of truth unless the user explicitly says to promote a point from it into project config or memory.

## Scientific Plotting

Scientific figures should be concise and easy to read. Use understandable nouns, short labels, and brief annotations.

Avoid unnecessary in-figure explanations, redundant comments, decorative complexity, and long titles. Include only the text needed to interpret the result.

## API And CLI Design

Future analysis functions should be modular and directly callable from Python. Keep command-line entry points as thin wrappers around reusable functions.

CLI versions can remain available, but do not optimize primarily for command-line ergonomics unless the user explicitly asks for it.
