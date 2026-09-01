"""I/O helpers for the RELEX Python implementation."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import numpy as np

from .core import RelexResult


REQUIRED_TOP_KEYS = {"system", "integration", "options", "states"}


def load_input_json(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return normalize_input(data, base_dir=path.parent)

def normalize_input(data: dict, base_dir: Path) -> dict:
    missing = REQUIRED_TOP_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Missing required keys: {', '.join(sorted(missing))}")

    system = data["system"]
    integration = data["integration"]
    options = data["options"]

    system.setdefault("iw", 0)
    system.setdefault("iout", 0)
    integration.setdefault("itot", 0)
    options.setdefault("iopw", 0)
    options.setdefault("iopnuc", 0)

    grid = data.get("grid", {})
    if "ngrid" not in grid:
        grid["ngrid"] = 200
    if grid["ngrid"] % 2 != 0:
        raise ValueError("grid.ngrid must be even")
    data["grid"] = grid

    if options["iopw"] == 1:
        opt = data.get("optical_potential", {})
        if "path" in opt:
            opt_path = Path(opt["path"])
            if not opt_path.is_absolute():
                opt_path = base_dir / opt_path
            opt["path"] = str(opt_path)
        data["optical_potential"] = opt

    # Ensure states are sorted by i
    states = data["states"]
    states_sorted = sorted(states, key=lambda s: int(s["i"]))
    data["states"] = states_sorted

    return data


def save_outputs(result: RelexResult, output_dir: str | Path) -> Dict[str, str]:
    """Write relex.out, relex2.out, and results.json. Returns paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "relex.out"
    out2_path = output_dir / "relex2.out"
    json_path = output_dir / "results.json"

    inputs = result.inputs
    system = inputs["system"]
    integration = inputs["integration"]
    options = inputs["options"]

    ap = system["ap"]
    zp = system["zp"]
    at = system["at"]
    zt = system["zt"]
    eca = system["eca"]
    iw = system.get("iw", 0)
    iout = system.get("iout", 0)
    nb = integration["nb"]
    accur = integration["accur"]
    iopw = options.get("iopw", 0)
    iopnuc = options.get("iopnuc", 0)
    ngrid = inputs.get("grid", {}).get("ngrid", 200)

    states = inputs["states"]
    matrix_elements = inputs.get("matrix_elements", [])
    deformations = inputs.get("nuclear_deformation", [])

    with out_path.open("w", encoding="utf-8") as f:
        f.write(" \n")
        f.write(" ****          Output of RELEX          ****\n")
        f.write(" \n")
        f.write(" Input parameters:\n\n")
        f.write(
            "  Nuclear mass and charge numbers: {ap:4.0f}  {zp:4.0f}  {at:4.0f}  {zt:4.0f}\n"
            .format(ap=ap, zp=zp, at=at, zt=zt)
        )
        f.write("  Laboratory energy/per nucleon [MeV]: {eca:.3f}\n".format(eca=eca))
        f.write("  Option for the excited nucleus [0 proj / 1 targ]: {iw}\n".format(iw=iw))
        f.write("  Integration parameters (NGRID,NB,ACCUR): {ngrid} {nb} {accur}\n".format(ngrid=ngrid, nb=nb, accur=accur))
        f.write("  Enter optical potential 0(no) 1(yes): {iopw}\n".format(iopw=iopw))
        f.write("  Compute nuclear excitation 0(no) 1(yes): {iopnuc}\n".format(iopnuc=iopnuc))
        f.write("  Output of statistical 0(no) 1(yes): {iout}\n\n".format(iout=iout))

        f.write("  Number of nuclear states = {nst}\n\n".format(nst=len(states)))
        f.write("  State     E [MeV]        Spin\n")
        for s in states:
            f.write("  {i:3d}   {ex:10.3f}   {spin:10.3f}\n".format(i=int(s["i"]), ex=float(s["ex"]), spin=float(s["spin"])))

        f.write("\n Red. matrix elem.  (E1, M1: fm.e),  (E2: fm^2.e)\n\n")
        f.write("                E1         E2         M1\n")
        for me in matrix_elements:
            f.write(
                "  {i:2d} --> {j:2d}  {e1:10.3f}  {e2:10.3f}  {m1:10.3f}\n"
                .format(
                    i=int(me["i"]),
                    j=int(me["j"]),
                    e1=float(me.get("mate1", 0.0)),
                    e2=float(me.get("mate2", 0.0)),
                    m1=float(me.get("matm1", 0.0)),
                )
            )

        if iopnuc == 1 and deformations:
            f.write("\n  Deform. Par. DELTE0, DELTE1, DELTE2\n")
            f.write("  Transit.      DELTE0     DELTE1     DELTE2\n")
            for d in deformations:
                f.write(
                    "  {j:2d}      {d0:10.3f} {d1:10.3f} {d2:10.3f}\n"
                    .format(
                        j=int(d["j"]),
                        d0=float(d.get("delte0", 0.0)),
                        d1=float(d.get("delte1", 0.0)),
                        d2=float(d.get("delte2", 0.0)),
                    )
                )

        if integration.get("itot", 0) == 1:
            f.write("\n Impact parameter, state, and occupation probabilities\n")
            for ib, bval in enumerate(result.b):
                for j in range(1, result.spinave.shape[0]):
                    if j == 1:
                        f.write("  {b:12.4g} {j:2d} {p:12.4g}\n".format(b=bval, j=j + 1, p=result.spinave[j, ib]))
                    else:
                        f.write("               {j:2d} {p:12.4g}\n".format(j=j + 1, p=result.spinave[j, ib]))

        f.write("\n Cross sections in mb\n")
        for state, cross in result.cross_sections.items():
            f.write("  State {s:3d} = {c:12.4g}\n".format(s=state, c=cross))

    # Statistical tensors
    with out2_path.open("w", encoding="utf-8") as f:
        if iout != 1:
            f.write("No statistical tensor output requested.\n")
        else:
            f.write("The angular distribution tensors STAT(J,KA,KAPPA)\n")
            f.write("J     KA   KAPPA       Real STAT         Imag STAT\n")
            for ib, stat in enumerate(result.stat_by_b):
                f.write(f"b={result.b[ib]}\n")
                nst = stat.shape[0]
                for j in range(1, nst):
                    for ka in (0, 2, 4):
                        for kappa in range(0, ka + 1):
                            val = stat[j, ka, kappa + 4]
                            if val != 0:
                                f.write(" {j:3d} {ka:3d} {kp:4d} {re:18.6e} {im:18.6e}\n".format(
                                    j=j + 1,
                                    ka=ka,
                                    kp=kappa,
                                    re=float(np.real(val)),
                                    im=float(np.imag(val)),
                                ))

    # JSON summary
    summary = {
        "cross_sections": result.cross_sections,
        "b": result.b.tolist(),
        "spinave": result.spinave.tolist(),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "relex_out": str(out_path),
        "relex2_out": str(out2_path),
        "results_json": str(json_path),
    }
