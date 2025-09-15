# Disable Streamlit's file watcher to avoid torch.classes watcher crash on Windows
import os
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import zipfile
import tempfile
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import torch
from sbi import utils as sbi_utils
from sbi import inference as sbi_inference
import streamlit as st
import traci
import random
import io
import multiprocessing
import time  # <-- IMPORTED FOR THE TIMER

# Try to resolve SUMO binaries if available
try:
    from sumolib import checkBinary as _checkBinary
except Exception:
    _checkBinary = None

# ---------------- SETTINGS ----------------
SUMO_GUI = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"
SUMO_BINARY = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
MAX_STEPS = 5_000_000

# ---------------- UTILITIES ----------------
def resolve_sumo_binary(gui: bool) -> str:
    explicit = SUMO_GUI if gui else SUMO_BINARY
    if explicit and os.path.isfile(explicit):
        return explicit
    if _checkBinary is not None:
        try:
            return _checkBinary("sumo-gui" if gui else "sumo")
        except Exception:
            pass
    return "sumo-gui" if gui else "sumo"

class _chdir:
    def __init__(self, path):
        self.path = path
        self._old = None
    def __enter__(self):
        self._old = os.getcwd()
        os.chdir(self.path)
    def __exit__(self, exc_type, exc, tb):
        try:
            os.chdir(self._old)
        except Exception:
            pass

def safe_traci_close():
    try:
        traci.close(False)
    except Exception:
        pass

# ---------------- OUTPUTS & PARAMS ----------------
def build_output_paths(out_dir: str, run_tag: str):
    paths = {
        "summary": os.path.join(out_dir, f"output_summary_{run_tag}.xml"),
        "tripinfo": os.path.join(out_dir, f"output_tripinfo_{run_tag}.xml"),
    }
    cli = ["--summary-output", paths["summary"], "--tripinfo-output", paths["tripinfo"]]
    return paths, cli

def discover_attrs_from_file(xml_path: str):
    attrs = set()
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for elem in root:
            attrs.update(elem.attrib.keys())
    except Exception:
        pass
    return sorted(list(attrs))

def aggregate_xml_attributes(xml_path: str, attrs: list, agg: str = "mean"):
    vals = {a: [] for a in attrs}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for elem in root:
            for a in attrs:
                v = elem.get(a)
                if v is not None:
                    try:
                        vals[a].append(float(v))
                    except Exception:
                        pass
    except Exception:
        pass
    out = {}
    for a, lst in vals.items():
        if len(lst) == 0:
            out[a] = np.nan
        else:
            if agg == "last":
                out[a] = lst[-1]
            else:
                out[a] = float(np.nanmean(lst))
    return out

def apply_params_after_start(theta: dict):
    tls_scale = theta.get("tls_duration_scale", 1.0)
    if tls_scale and abs(tls_scale - 1.0) > 1e-6:
        try:
            tls_ids = traci.trafficlight.getIDList()
            for tls_id in tls_ids:
                logics = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
                new_logics = []
                for logic in logics:
                    phases = []
                    for ph in logic.phases:
                        dur = max(1.0, ph.duration * tls_scale)
                        min_d = max(1.0, ph.minDur * tls_scale if ph.minDur is not None else dur)
                        max_d = max(min_d, ph.maxDur * tls_scale if ph.maxDur is not None else dur)
                        phases.append(traci.trafficlight.Phase(dur, ph.state, minDur=min_d, maxDur=max_d))
                    logic.phases = phases
                    new_logics.append(logic)
                for logic in new_logics:
                    traci.trafficlight.setCompleteRedYellowGreenDefinition(tls_id, logic)
        except Exception:
            pass

    speed_factor = theta.get("veh_speed_factor", 1.0)
    if speed_factor and abs(speed_factor - 1.0) > 1e-6:
        try:
            vtypes = traci.vehicletype.getIDList()
            for vt in vtypes:
                try:
                    traci.vehicletype.setSpeedFactor(vt, float(speed_factor))
                except Exception:
                    pass
        except Exception:
            pass

def run_sumo_collect(cfg_file: str, gui: bool, out_dir: str, run_tag: str, theta: dict):
    binary = resolve_sumo_binary(gui)
    cfg_abs = os.path.abspath(cfg_file)
    cfg_dir = os.path.dirname(cfg_abs)
    cfg_base = os.path.basename(cfg_abs)
    out_paths, out_cli = build_output_paths(out_dir, run_tag)
    safe_traci_close()
    try:
        with _chdir(cfg_dir):
            cli = [binary, "-c", cfg_base, "--no-step-log", "true", "--quit-on-end"]
            if "demand_scale" in theta and theta["demand_scale"] is not None:
                cli += ["--scale", str(float(theta["demand_scale"]))]
            cli += out_cli
            traci.start(cli)
            apply_params_after_start(theta)
            steps = 0
            while traci.simulation.getMinExpectedNumber() > 0 and steps < MAX_STEPS:
                traci.simulationStep()
                steps += 1
            safe_traci_close()
        return out_paths, True
    except Exception:
        safe_traci_close()
        return out_paths, False

# ---------------- SBI ----------------
def train_sbi(theta_df: pd.DataFrame, x_df: pd.DataFrame, seed: int = 0):
    torch.manual_seed(seed)
    d_theta = theta_df.shape[1]
    prior = sbi_utils.BoxUniform(low=torch.zeros(d_theta), high=torch.ones(d_theta))
    theta_vals = theta_df.values.astype(np.float32)
    theta_min = theta_vals.min(axis=0, keepdims=True)
    theta_max = theta_vals.max(axis=0, keepdims=True)
    denom = np.clip(theta_max - theta_min, 1e-8, None)
    theta_norm = (theta_vals - theta_min) / denom
    x_vals = x_df.values.astype(np.float32)
    inference = sbi_inference.SNPE(prior=prior)
    density_estimator = inference.append_simulations(
        torch.tensor(theta_norm, dtype=torch.float32),
        torch.tensor(x_vals, dtype=torch.float32)
    ).train()
    posterior = inference.build_posterior(density_estimator)
    return posterior, (theta_min, theta_max)

# ---------------- MULTIPROCESSING WORKER FUNCTION ----------------
def run_simulation_worker(args):
    """A self-contained function to be executed by each parallel worker."""
    i, cfg_file, out_dir, selected_attrs, demand_range, tls_dur_range, speed_fac_range = args
    run_tag = f"train_{i+1}"
    
    theta = {
        "demand_scale": random.uniform(*demand_range),
        "tls_duration_scale": random.uniform(*tls_dur_range),
        "veh_speed_factor": random.uniform(*speed_fac_range),
    }

    paths, ok = run_sumo_collect(
        cfg_file=cfg_file, gui=False, out_dir=out_dir, run_tag=run_tag, theta=theta
    )

    if not ok:
        return None
    
    x_vec = {}
    x_vec.update(aggregate_xml_attributes(paths["summary"], selected_attrs["summary"]))
    x_vec.update(aggregate_xml_attributes(paths["tripinfo"], selected_attrs["tripinfo"]))
    
    return theta, x_vec

# ---------------- STREAMLIT UI ----------------
def main():
    st.title("SUMO + SBI Parameter-Effect Trainer (Parallelized)")

    if 'work_dir' not in st.session_state:
        st.session_state.work_dir = None
        st.session_state.cfg_file = None
        st.session_state.probe_paths = None
        st.session_state.selected_attrs = {"summary": [], "tripinfo": []}

    uploaded_zip = st.file_uploader("Upload your SUMO scenario (ZIP)", type="zip")

    if uploaded_zip and st.session_state.work_dir is None:
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue()), "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        st.session_state.work_dir = temp_dir
        st.success(f"Unzipped scenario to temporary folder.")

    if st.session_state.work_dir:
        cfg_files = [os.path.join(r, f) for r, _, fs in os.walk(st.session_state.work_dir) for f in fs if f.lower().endswith(".sumocfg")]
        if not cfg_files: st.error("No .sumocfg file found in ZIP."); st.stop()
        
        default_cfg = next((f for f in cfg_files if "osm.sumocfg" in f.lower()), cfg_files[0])
        st.session_state.cfg_file = st.selectbox("Select SUMO config", cfg_files, index=cfg_files.index(default_cfg))

        st.markdown("---")
        st.header("Step 1: Generate probe outputs")
        if st.button("Generate Probe Outputs"):
            with st.spinner("Running single SUMO simulation..."):
                paths, ok = run_sumo_collect(st.session_state.cfg_file, False, st.session_state.work_dir, "probe", {})
            if ok: st.success("Probe run completed."); st.session_state.probe_paths = paths
            else: st.error("Probe run failed."); st.stop()

        if st.session_state.probe_paths:
            st.subheader("Select output attributes for 'x'")
            sum_attrs = discover_attrs_from_file(st.session_state.probe_paths["summary"])
            tri_attrs = discover_attrs_from_file(st.session_state.probe_paths["tripinfo"])
            st.session_state.selected_attrs["summary"] = st.multiselect("Summary attributes", sum_attrs, default=st.session_state.selected_attrs.get("summary"))
            st.session_state.selected_attrs["tripinfo"] = st.multiselect("Tripinfo attributes", tri_attrs, default=st.session_state.selected_attrs.get("tripinfo"))

            st.markdown("---")
            st.header("Step 2: Configure and run training")
            demand_range = st.slider("Demand Scale", 0.1, 5.0, (0.5, 1.5))
            tls_dur_range = st.slider("TLS Duration Scale", 0.1, 5.0, (0.5, 2.0))
            speed_fac_range = st.slider("Vehicle Speed Factor", 0.1, 2.0, (0.8, 1.2))

            num_sims = st.number_input("Number of simulations (N)", 10, 10000, 100)
            
            max_workers = os.cpu_count() or 1
            num_workers = st.slider("Number of parallel workers", 1, max_workers, max(1, max_workers - 1))

            if st.button("Run Simulations & Train SBI Model"):
                start_time = time.time()  # <-- START TIMER

                if not st.session_state.selected_attrs["summary"] and not st.session_state.selected_attrs["tripinfo"]:
                    st.error("Please select at least one output attribute."); st.stop()
                
                tasks = [(i, st.session_state.cfg_file, st.session_state.work_dir, st.session_state.selected_attrs, demand_range, tls_dur_range, speed_fac_range) for i in range(num_sims)]
                
                st.info(f"Starting {num_sims} simulations across {num_workers} parallel processes...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                with multiprocessing.Pool(processes=num_workers) as pool:
                    for i, result in enumerate(pool.imap_unordered(run_simulation_worker, tasks)):
                        if result:
                            results.append(result)
                        status_text.text(f"Completed {i+1}/{num_sims} simulations... ({len(results)} successful)")
                        progress_bar.progress((i + 1) / num_sims)
                
                status_text.text(f"All simulations complete. {len(results)} succeeded.")
                if not results: st.error("All simulation runs failed."); st.stop()

                all_theta, all_x = zip(*results)
                theta_df = pd.DataFrame(list(all_theta))
                x_df = pd.DataFrame(list(all_x)).fillna(0)

                with st.spinner("Training SBI model..."):
                    posterior, _ = train_sbi(theta_df, x_df)
                    st.session_state.posterior = posterior
                
                end_time = time.time()  # <-- END TIMER
                elapsed_time = end_time - start_time

                st.success(f"🎉 SBI model training complete!")
                st.info(f"Total time elapsed: {elapsed_time:.2f} seconds") # <-- DISPLAY TIME
                
                st.dataframe(theta_df.head())
                st.dataframe(x_df.head())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()