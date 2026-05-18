import torch

from src.filtrations.base import Diagram
from src.registry import FILTRATIONS


@FILTRATIONS("combined")
def combined(params=None):
    """Concatenate several filtrations into one diagram.

    Each sub-filtration's points are tagged with a unique marker so the model
    and `idx`-filtering at collate time can distinguish them:
      - column 3 (sublevel)     gets ``+ k * sublevel_offset`` added (default 10)
      - column 5 (direction_idx) gets ``+ k * idx_offset``  added (default 100)
    where k is the filtration's position in the list.

    Example::

      filtration:
        name: combined
        args:
          filtrations:
            - name: pht_directional
              args: {agg: add}
            - name: edt_sublevel
              args: {phase: pore}

    With defaults pht direction_idx stays 0..16, edt direction_idx becomes 100/101.
    Set ``filtration.diagram_idx: null`` to keep all points (recommended), or use
    explicit lists like ``[0,2,4,6,7,9,11,13,16,100]``.
    """
    cfg = dict(params or {})
    parts = cfg.get("filtrations") or []
    if not parts:
        raise ValueError("'combined' requires a non-empty 'filtrations' list")
    idx_offset = int(cfg.get("idx_offset", 100))
    sublevel_offset = float(cfg.get("sublevel_offset", 10.0))

    fns = []
    for spec in parts:
        if not isinstance(spec, dict) or "name" not in spec:
            raise ValueError("each filtration spec must be a dict with a 'name' key")
        sub_args = dict(spec.get("args") or {})
        fns.append(FILTRATIONS.get(spec["name"])(sub_args))

    def apply(image):
        out_chunks = []
        for k, fn in enumerate(fns):
            dgm = fn(image)
            pts = dgm.points
            if pts.shape[0] == 0:
                continue
            if pts.shape[1] != 6:
                raise ValueError(
                    "combined: sub-filtration must return 6-column points, got "
                    f"shape {tuple(pts.shape)}"
                )
            pts = pts.clone().to(torch.float32)
            pts[:, 3] = pts[:, 3] + float(k) * sublevel_offset
            pts[:, 5] = pts[:, 5] + float(k) * float(idx_offset)
            out_chunks.append(pts)
        if not out_chunks:
            points = torch.zeros((0, 6), dtype=torch.float32)
        else:
            points = torch.cat(out_chunks, dim=0)
        return Diagram(
            points=points,
            schema=["birth", "death", "dim", "sublevel", "direction_alpha", "direction_idx"],
        )

    return apply
