# station-viewer patches

Our changes to `software/station/clients/station-viewer` in
[norma-core](https://github.com/norma-core/norma-core) (restyle, RealSense 3D view, agent panel, IK/arm pose).

- `0001`–`0004`: commits on branch `devin/station-viewer-restyle` (made with `git format-patch`).
- `0005`: the uncommitted work on top of them, as of 2026-11-22.

Apply on a clean norma-core checkout:

```sh
cd norma-core
git checkout 6b494f7            # the upstream main these were based on
git am ../normacore-mcp/station-viewer-patches/000[1-4]-*.patch
git apply ../normacore-mcp/station-viewer-patches/0005-*.patch
```
