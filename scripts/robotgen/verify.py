#!/usr/bin/env python3
"""Check a generated robot against pinocchio and write the results to verify.json.

    verify.py NAME --repo REPO --vamp VAMP_SRC --pdqsort DIR --work DIR --cache DIR [--samples N]

1. Every sphere center and radius of the VAMP kernel matches pinocchio's placement of the
   same sphere in the whole-body URDF.
2. The kernel's self-collision result in an empty environment matches a pinocchio check of
   the sphere model with the generated SRDF.
3. The generated CRBA equals A^T M(A r + b) A, computed by pinocchio on the uncoupled arm
   model, where A and b collect the URDF mimic relations.
4. Every link of the whole-body model has the pose it has in the unmodified upstream URDF,
   with the base pose applied, each nested arm segment at its share of the extension and
   each held joint at its recipe position. A recipe frame, such as a tool center point,
   sits at its offset from its parent link.
5. The report holds the sphere fit (sphere count per link, the largest distance by which a
   mesh point lies outside the spheres, and the volume the spheres add outside the meshes).

Run it inside the scripts/robotgen pixi environment.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pinocchio as pin

HERE = Path(__file__).resolve().parent


def compile_kernel_harness(recipe, repo, vamp, pdqsort, work) -> Path:
    exe = work / f"verify_{recipe['name']}"
    header = repo / "include/geodex/integration/vamp/robots/generated" / f"{recipe['name']}.hh"
    eigen = Path(sys.prefix) / "include" / "eigen3"
    cmd = ["c++", "-std=c++17", "-O1", "-mavx2", "-mfma", "-Wno-ignored-attributes",
           f"-DKERNEL_HEADER=\"{header}\"", f"-DKERNEL_STRUCT={recipe['vamp_struct']}",
           f"-I{vamp / 'src' / 'impl'}", f"-I{pdqsort}", f"-I{eigen}",
           str(HERE / "verify_kernel.cpp"), "-o", str(exe)]
    subprocess.run(cmd, check=True)
    return exe


def kernel_spheres(exe: Path, qs: np.ndarray):
    text = "\n".join(" ".join(repr(float(v)) for v in q) for q in qs) + "\n"
    out = subprocess.run([str(exe)], input=text, capture_output=True, text=True, check=True).stdout
    valid, spheres, cur = [], [], None
    for line in out.splitlines():
        if line.startswith("valid"):
            valid.append(line.split()[1] == "1")
            cur = []
            spheres.append(cur)
        else:
            cur.append([float(v) for v in line.split()])
    return np.array(valid), np.array(spheres)


def pin_spheres(model, geom, data, q):
    gdata = pin.GeometryData(geom)
    pin.forwardKinematics(model, data, q)
    pin.updateGeometryPlacements(model, data, geom, gdata)
    return np.array([[*gdata.oMg[i].translation, geom.geometryObjects[i].geometry.radius]
                     for i in range(geom.ngeoms)])


def pin_self_valid(model, geom, q) -> bool:
    data, gdata = model.createData(), pin.GeometryData(geom)
    return not pin.computeCollisions(model, data, geom, gdata, q, True)


def coupling(urdf: Path, model) -> tuple[np.ndarray, np.ndarray, list[str]]:
    mimic = {}
    for j in ET.parse(urdf).getroot().findall("joint"):
        m = j.find("mimic")
        if m is not None:
            mimic[j.get("name")] = (m.get("joint"), float(m.get("multiplier", 1)),
                                    float(m.get("offset", 0)))
    names = [model.names[i] for i in range(1, model.njoints)]
    independent = [n for n in names if n not in mimic]
    A = np.zeros((model.nq, len(independent)))
    b = np.zeros(model.nq)
    for i, n in enumerate(names, start=1):
        iq = model.idx_qs[i]
        if n in mimic:
            primary, mult, off = mimic[n]
            A[iq, independent.index(primary)] = mult
            b[iq] = off
        else:
            A[iq, independent.index(n)] = 1.0
    return A, b, independent


def crba_name(recipe) -> str:
    return recipe.get("crba", {}).get("name", recipe["name"])


def crba_library(recipe, repo, work) -> ctypes.CDLL:
    src = repo / "src/robots/generated" / f"{crba_name(recipe)}_crba.cpp"
    lib = work / f"lib{crba_name(recipe)}_crba.so"
    subprocess.run(["c++", "-O2", "-shared", "-fPIC", "-ffp-contract=off", str(src), "-o",
                    str(lib)], check=True)
    return ctypes.CDLL(str(lib))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--vamp", type=Path, required=True)
    parser.add_argument("--pdqsort", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads((HERE / "robots" / f"{args.name}.json").read_text())
    repo, work = args.repo.resolve(), (args.work / args.name).resolve()
    work.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    import robotgen

    data_dir = robotgen.data_dir(recipe, repo)
    planar = "drive" in recipe
    report = {"robot": args.name, "samples": args.samples, "seed": 7}
    rng = np.random.default_rng(7)

    # Whole-body sphere model.
    urdf = data_dir / f"{args.name}_spherized.urdf"
    model = pin.buildModelFromUrdf(str(urdf), mimic=True)
    geom = pin.buildGeomFromUrdf(model, str(urdf), pin.GeometryType.COLLISION)
    lo, hi = model.lowerPositionLimit.copy(), model.upperPositionLimit.copy()
    if planar:
        lo[:2], hi[:2] = -5.0, 5.0
    qs = lo + (hi - lo) * rng.random((args.samples, model.nq))
    exe = compile_kernel_harness(recipe, repo, args.vamp, args.pdqsort, work)
    valid, spheres = kernel_spheres(exe, qs)
    data = model.createData()
    err_c = err_r = 0.0
    for q, ks in zip(qs, spheres):
        ps = pin_spheres(model, geom, data, q)
        err_c = max(err_c, float(np.abs(ks[:, :3] - ps[:, :3]).max()))
        err_r = max(err_r, float(np.abs(ks[:, 3] - ps[:, 3]).max()))
    report["fk_max_center_error_m"] = err_c
    report["fk_max_radius_error_m"] = err_r

    geom_self = pin.buildGeomFromUrdf(model, str(urdf), pin.GeometryType.COLLISION)
    geom_self.addAllCollisionPairs()
    pin.removeCollisionPairs(model, geom_self, str(data_dir / f"{args.name}.srdf"), False)
    ref = np.array([pin_self_valid(model, geom_self, q) for q in qs])
    report["self_collision_agreement"] = float((ref == valid).mean())
    report["kernel_self_collision_free_fraction"] = float(valid.mean())

    # Coupled CRBA against pinocchio on the uncoupled dynamics model.
    dyn = data_dir / f"{args.name}_dynamics.urdf"
    full = pin.buildModelFromUrdf(str(dyn))
    A, b, independent = coupling(dyn, full)
    lib = crba_library(recipe, repo, work)
    fn = getattr(lib, f"{crba_name(recipe)}_crba")
    n = A.shape[1]
    fn.argtypes = [np.ctypeslib.ndpointer(np.float64), np.ctypeslib.ndpointer(np.float64)]
    rlo = np.array([full.lowerPositionLimit[full.idx_qs[full.getJointId(j)]] for j in independent])
    rhi = np.array([full.upperPositionLimit[full.idx_qs[full.getJointId(j)]] for j in independent])
    fdata = full.createData()
    err_m = 0.0
    scale = 0.0
    for _ in range(args.samples):
        r = rlo + (rhi - rlo) * rng.random(n)
        M = pin.crba(full, fdata, A @ r + b)
        M = np.triu(M) + np.triu(M, 1).T
        ref_m = A.T @ M @ A
        out = np.zeros(n * (n + 1) // 2)
        fn(np.ascontiguousarray(r), out)
        gen = np.zeros((n, n))
        k = 0
        for i in range(n):
            for j in range(i, n):
                gen[i, j] = gen[j, i] = out[k]
                k += 1
        err_m = max(err_m, float(np.abs(gen - ref_m).max()))
        scale = max(scale, float(np.abs(ref_m).max()))
    report["crba_max_abs_error"] = err_m
    report["crba_max_entry"] = scale
    report["crba_coordinates"] = independent

    # Link and end-effector poses against the unmodified upstream description.
    raw, _ = robotgen.load_source_urdf(recipe, args.cache.resolve())
    for link in raw.findall("link"):
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                link.remove(el)
    up = pin.buildModelFromXML(ET.tostring(raw, encoding="unicode"))
    up_data = up.createData()
    frames = {f["name"]: f for f in recipe.get("frames", [])}
    ee = recipe["end_effector"]
    ee_offset = pin.SE3.Identity()
    if ee in frames:
        ee_offset = pin.SE3(pin.rpy.rpyToMatrix(*frames[ee]["rpy"]), np.array(frames[ee]["xyz"]))
        ee = frames[ee]["parent"]
    ee_up = up.getFrameId(ee)
    ee_wb = model.getFrameId(recipe["end_effector"])
    links = [f.name for f in model.frames if f.type == pin.FrameType.BODY and up.existFrame(f.name)]
    held = dict(recipe.get("joint_positions", {}))
    for j in raw.findall("joint"):
        m = j.find("mimic")
        if m is not None and m.get("joint") in recipe.get("joint_positions", {}):
            held[j.get("name")] = (float(m.get("multiplier", 1)) *
                                   recipe["joint_positions"][m.get("joint")] +
                                   float(m.get("offset", 0)))
    shares = {}
    for c in recipe.get("couplings", []):
        uppers = [float(next(j for j in raw.findall("joint") if j.get("name") == s)
                        .find("limit").get("upper")) for s in c["segments"]]
        for s, u in zip(c["segments"], uppers):
            shares[s] = (c["joint"], u / sum(uppers))
    wb_names = [model.names[i] for i in range(1, model.njoints) if model.joints[i].nq > 0]
    err_ee = err_links = 0.0
    for q in qs[: min(200, len(qs))]:
        qu = pin.neutral(up)
        for i in range(1, up.njoints):
            name = up.names[i]
            if up.joints[i].nq != 1:
                continue
            if name in shares:
                primary, frac = shares[name]
                qu[up.idx_qs[i]] = frac * q[wb_names.index(primary)]
            elif name in wb_names:
                qu[up.idx_qs[i]] = q[wb_names.index(name)]
            elif name in held:
                qu[up.idx_qs[i]] = held[name]
        pin.framesForwardKinematics(up, up_data, qu)
        pin.framesForwardKinematics(model, data, q)
        base = pin.SE3.Identity()
        if planar:
            base = pin.SE3(pin.rpy.rpyToMatrix(0.0, 0.0, q[2]),
                           np.array([q[0], q[1], recipe.get("base_height", 0.0)]))
        diff = (base * up_data.oMf[ee_up] * ee_offset).homogeneous - data.oMf[ee_wb].homogeneous
        err_ee = max(err_ee, float(np.abs(diff).max()))
        for link in links:
            diff = ((base * up_data.oMf[up.getFrameId(link)]).homogeneous -
                    data.oMf[model.getFrameId(link)].homogeneous)
            err_links = max(err_links, float(np.abs(diff).max()))
    report["upstream_ee_max_error"] = err_ee
    report["upstream_link_max_error"] = err_links
    report["upstream_links_compared"] = len(links)

    fit = robotgen.sphere_fit(args.work.resolve() / args.name / f"{args.name}_mesh.urdf", urdf)
    report["sphere_fit"] = {"links": fit["links"], "total": fit["total"]}
    print(json.dumps(report, indent=2))
    (work / "verify.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
