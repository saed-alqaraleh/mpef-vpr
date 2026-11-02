# Internal release audit gates — 2026-11-02

| Gate | Status | Evidence |
| --- | --- | --- |
| 21-feature source code | Passed locally | Recovered typed evidence and feature implementation |
| Frozen fitted models and calibrators | Passed locally | Four original SHA-256 byte matches |
| Ranking vs calibrated probability separation | Passed locally | \`src/mpef_vpr/frozen.py\` |
| CPU installation, tests, retrieval audit | Passed locally | Sixteen tests, frozen checks and query-mean replay |
| Tokyo 147-query retrieval-only verification | Passed locally | 294 per-query result rows; 16,140 gallery |
| GitHub private staging | Created | \`saed-alqaraleh/mpef-vpr\` |
| GitHub remote CI | Must inspect successful run | Workflow committed; local pass alone does not imply GitHub run pass |
| Full original six-domain raw-prediction replay | Not established | Original candidate-level archives not distributed here |
| Modern zero-refit SALAD–ALIKED 3-domain transfer | Not established | Raw matching evidence and paired predictions absent here |
| Original image/matcher pipeline recreation | Not established | Upstream datasets and checkpoints separately sourced |
| Original collaborator source redistribution license | Blocked for public | Source rights notice says all rights reserved |
| Fitted models and derived-result distribution | Requires authorization | Rights and benchmark conditions unresolved |
| Final manuscript equivalence | Partially reviewed | Confirm latest manuscript/citations before public release |
| Public GitHub release and DOI | Not authorized | Keep repository private |

Do not switch to Public or add MIT/Apache/BSD licensing until scientific, legal and authorship approvals are explicitly completed.
