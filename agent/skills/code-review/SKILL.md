---
name: code-review
description: Review a code change for correctness, regressions, security, and missing tests
tools: ["grep", "read_path", "path_status"]
---
# Code review

Inspect the changed code and the smallest relevant set of callers, tests, and contracts.
Use semantic language-server navigation when that optional tool is enabled for the active
agent profile; otherwise use exact workspace search and do not treat its absence as a blocker.
Prioritize concrete correctness, security, data-loss, concurrency, and compatibility issues
over stylistic preferences. Ground each finding in observed code and identify the exact path
and location. Check whether existing tests cover the risky behavior. Do not edit files unless
the user separately asks for fixes. If no material issue is found, say so and state any testing
or runtime evidence that was unavailable.
