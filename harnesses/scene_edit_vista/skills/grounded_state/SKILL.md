# Grounded state

Invoke this preparation skill for every new request, before language parsing.
The bound handler calls inspect once and returns current public facts. An empty
successful observation is authoritative. Observation failure is distinct from an
empty scene; the policy may then use supplied current request facts with explicit
provenance. Never recover facts from an earlier request's cached scene.

Use support IDs, room, world bounds and position as factual context. Names do not
establish physical support or identity. Collision is a host fact, not proof of an
invalid placement. Geometry and commit validation remain with the host.

The policy joins confirmed insertion records to this observation by stable_id.
A unique match recovers the current name, room and support. Missing or duplicated
identities remain unresolved, without asset, proximity or old-name fallback.
Deleted identities must never be rebound to replacement objects. History remains
unchanged so historical placement and current location remain distinguishable.
