# Reliability Agent Addendum

This repository's framing/chunking/ecc work is owned by the Reliability Agent.

## Definition of Done
- Packets must remain forward-compatible via versioned JSON framing (include `msg_id`, `seq`, `total`, and `cfg`).
- Chunking must allow independent reassembly with or without CRC/ECC enabled.
- Fault-injection harnesses and automated tests must exist for ECC/CRC behaviours.
- CLI options must expose chunking, CRC, and ECC toggles with clear messaging about optional dependencies.

## Guardrails
- ECC is feature-flagged; both ECC-enabled and ECC-disabled paths must pass tests.
- Secrets (keys, plaintext messages) must never be logged.
- `msg_id`, `seq`, `total`, and `cfg` are mandatory in any serialized packet or JSON output.

