# Building the integrity release

Branch pushes and pull requests run lightweight integrity tests. They do not
publish images. The separate release workflow requires manual dispatch and an
explicit publish choice; it never updates `latest` automatically.

Build all six repositories using the source lock and `scripts/build_pipeline_images.py`
in the companion mewc-infrastructure integration branch. The builder records
immutable local image IDs, exact source commits and frozen runtime base images.
Use its generated stage image lock for pipeline runs. Keep the image archive
when transporting locally built images to another host.

The detector and flow Dockerfiles extend digest-pinned previous runtime images
and replace their application source. This preserves the tested ML environment
while fixing the stage contracts. It is not a claim that every inherited runtime
dependency is current or free of vulnerabilities. Training/runtime upgrades are
separate from this inference integrity repair.

Snip, EXIF and box require `--build-arg MEWC_DETECT_BASE=<fixed-parent>`; predict
requires `--build-arg MEWC_FLOW_BASE=<fixed-parent>`. Use the newly fixed parent,
not an older published stage tag. The manual release workflow requires a
published parent digest. Local integration builds use the recorded parent image
ID through a temporary local tag. Publish compatible parents before children,
then update the deployment image lock with the returned registry digests.

Public tests use synthetic inputs and mocked ML dispatch. Actual inference and
private-model comparisons are documented in the integration validation record.

Manual release builds (including build-only dispatches) require an exact match to
`release-parent.json`, restricted to the declared `zaandahl` parent repository.
The `image` field is deliberately unset because this review has local image IDs,
not published registry digests. After publishing the parent from `source_commit`,
a maintainer must verify its provenance and commit its `repository@sha256` digest
to this lock before dispatching a child release. Digest syntax alone is insufficient.
This approval record does not independently attest how the parent was built.

For Compose builds, export `MEWC_FLOW_BASE` with the fixed parent
used by the source builder, then run `docker compose build`. The variable is required.
