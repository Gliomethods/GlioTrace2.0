from typing import Tuple
from pathlib import Path
from typing import Optional
import pandas as pd
import json
import numpy as np
import joblib

from tqdm import tqdm

from gliotrace.build_tracks_and_vascularity.weights_and_models.load_networks import load_trained_networks

from gliotrace.initalize_class.load_data import build_stack_table_flex
from gliotrace.build_tracks_and_vascularity.build_tracks_and_vascularity import build_tracks_and_vascularity
from gliotrace.feature_and_hmm_pipline.hmm_pipeline import hmm_pipeline
from gliotrace.visualize.generate_video import generate_video
from gliotrace.visualize.generate_video_compare import generate_video_compare

from gliotrace.initalize_class.defaults import SOFTMAX_COLUMNS

from gliotrace.initalize_class.validation import validate_init, validate_fcols, validate_hmm_param, validate_detection_sensitivity


PathLike = str | Path


class Gliotrace:
    """
      1) building a stack table from stackfiles + metadata
      2) tracking + vascularity computation
      3) HMM/GLM pipeline
      4) video generation
    """

    def __init__(
        self,
        stackfile,
        metadata,
        detection_sensitivity: float = 0.0,
        channel_roles: dict[str, str] | None = None,
        fcols: list[str] | None = None,
        hmm_param: dict[str, int | float] = None,
        control: str = "all",
        patient_id: Optional[list[str]] = None,
        sets_by_patient: Optional[dict[str, list[str]]] = None,
        treatment: list[str] = None,
        verbose: bool = False,
    ):
        v = validate_init(
            stackfile=stackfile,
            metadata=metadata,
            detection_sensitivity=detection_sensitivity,
            channel_roles=channel_roles,
            fcols=fcols,
            hmm_param=hmm_param,
            control=control,
            patient_id=patient_id,
            sets_by_patient=sets_by_patient,
            treatment=treatment,
            verbose=verbose,
        )

        self.verbose = v.verbose

        self.control = control
        # Build stacktable containing all informaiton about the run data
        self.subtable = build_stack_table_flex(
            v.stackfiles,
            v.metadata,
            v.treatment,
            v.control,
            v.patient_id,
            v.sets_by_patient)

        # Start off with all stacks being untracked
        self.subtable["tracked"] = False

        # Store validated config
        self.detection_sensitivity = v.detection_sensitivity
        self.channel_roles = v.channel_roles
        self.fcols = v.fcols
        self.hmm_param = v.hmm_param

        # Internal state
        self._tracked = False
        self._hmm = False

        # Keep track of output paths
        self._generate_video_visualizations = []

        # Empty track data
        self.track_data = None

        if self.verbose:
            self._print_configuration()

    # ------------------------------------------------------------------
    # Guards / internal helpers
    # ------------------------------------------------------------------

    def _require_tracked(self):
        if not self._tracked:
            raise RuntimeError(
                "run_tracking must be called before this method")

    def _require_hmm(self):
        if not self._hmm:
            raise RuntimeError("glm_hmm must be called before this method")

    def patients(self, treated: bool | None = None) -> list[str]:
        st = self.subtable
        if "patient_id" not in st.columns:
            raise ValueError("stacktable has no 'patient_id' column")

        if treated is not None:
            if "is_treatment" not in st.columns:
                raise ValueError(
                    "treated filter requested but stacktable has no 'is_treatment' column")
            st = st[st["is_treatment"] == bool(treated)]

        return sorted(st["patient_id"].dropna().astype(str).unique().tolist())

    def sets(self, patient: str, treated: bool | None = None) -> list[int]:
        st = self.subtable
        if "patient_id" not in st.columns:
            raise ValueError("stacktable has no 'patient_id' column")
        if "set" not in st.columns:
            raise ValueError("stacktable has no 'set' column")

        st = st[st["patient_id"].astype(str) == str(patient)]
        if st.empty:
            raise ValueError(f"Unknown patient_id={patient}")

        if treated is not None:
            if "is_treatment" not in st.columns:
                raise ValueError(
                    "treated filter requested but stacktable has no 'is_treatment' column")
            st = st[st["is_treatment"] == bool(treated)]

        return sorted(st["set"].dropna().astype(int).unique().tolist())

    def exps(self, patient: str | None = None, set: int | None = None, treated: bool | None = None) -> list[int]:
        st = self.subtable

        # patient filter (optional)
        if patient is not None:
            if "patient_id" not in st.columns:
                raise ValueError("stacktable has no 'patient_id' column")
            st = st[st["patient_id"].astype(str) == str(patient)]
            if st.empty:
                raise ValueError(f"Unknown patient_id={patient}")

        # set filter (ONLY allowed if patient was provided)
        if set is not None:
            if patient is None:
                raise ValueError(
                    "set was provided but patient is None; set is only valid within a patient")
            if "set" not in st.columns:
                raise ValueError("stacktable has no 'set' column")
            st = st[st["set"] == int(set)]
            if st.empty:
                raise ValueError(
                    f"No experiments for patient_id={patient} and set={set}")

        # treated filter (optional)
        if treated is not None:
            if "is_treatment" not in st.columns:
                raise ValueError(
                    "treated filter requested but stacktable has no 'is_treatment' column")
            st = st[st["is_treatment"] == bool(treated)]

        return sorted(st["exp"].dropna().astype(int).unique().tolist())

    def _validate_exp_roi(self, exp, roi):
        try:
            exp = int(exp)
        except Exception:
            raise TypeError("exp must be int-like")

        try:
            roi = int(roi)
        except Exception:
            raise TypeError("roi must be int-like")

        exp_list = sorted(set(self.track_data["exp"].astype(int).unique()))
        if exp not in exp_list:
            raise ValueError(f"exp={exp} not found. Available exp: {exp_list}")

        roi_list = sorted(
            set(
                self.track_data.loc[self.track_data["exp"].astype(
                    int) == exp, "roi"]
                .astype(int)
                .unique()
            )
        )
        if roi not in roi_list:
            raise ValueError(
                f"roi={roi} not found for exp={exp}. Available roi: {roi_list}")

        return exp, roi

    def assert_same_rows_except_tracked(self, current: pd.DataFrame, loaded: pd.DataFrame):
        # drop tracked if present
        c = current.drop(columns=["tracked"], errors="ignore").copy()
        l = loaded.drop(columns=["tracked"], errors="ignore").copy()

        # normalize index + column order for a fair comparison
        c = c.reset_index(drop=True)
        l = l.reset_index(drop=True)

        c = c.reindex(sorted(c.columns), axis=1)
        l = l.reindex(sorted(l.columns), axis=1)

        if c.shape != l.shape:
            raise ValueError(
                f"Subtable mismatch: shape current={c.shape}, loaded={l.shape}")

        if not c.equals(l):
            # show a small hint of where it differs
            diff = (c != l) & ~(c.isna() & l.isna())
            first = diff.stack()
            first = first[first].index.tolist()
            where = first[0] if first else None
            raise ValueError(
                f"Subtable mismatch (ignoring 'tracked'). First difference at {where}")

    def _coerce_output_path(self, output: Optional[PathLike], exp: int, roi: int):
        """
        Decide where to write output.

        - If output is None, write to ./gliotrace_videos/exp_{exp}/roi_{roi}
        - If output is a directory, create it
        - If output is a file path, create its parent directory
        """
        if output is None:
            out = Path.cwd() / "gliotrace_videos" / f"exp_{exp}" / f"roi_{roi}"
        else:
            out = Path(output)

        # If they passed something with a suffix, treat as file path; else dir.
        if out.suffix:
            out.parent.mkdir(parents=True, exist_ok=True)
        else:
            out.mkdir(parents=True, exist_ok=True)

        return out

    def _prepare_gbm_vasc_arrays(self, gbm_array: np.ndarray, vasc_array: np.ndarray, stack_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize GBM/VASC arrays to (H, W, T), validate uint8-castability, cast to uint8,
        and crop spatial dims to be divisible by 8.
        """
        stack_path = Path(stack_path)

        # Make sure stack is of dimensions width x height x time (works also for 500 x 501 x time)
        if not abs(gbm_array.shape[0] - gbm_array.shape[1]) < 5:
            gbm_array = np.moveaxis(gbm_array, 0, -1)
            vasc_array = np.moveaxis(vasc_array, 0, -1)

        if not (np.all(np.isfinite(gbm_array)) and
                np.all((gbm_array >= 0) & (gbm_array <= 255)) and
                np.all(np.equal(gbm_array, np.round(gbm_array)))):
            raise ValueError(
                f"In {stack_path}: gbm images must be safely castable to uint8")

        gbm = gbm_array.astype(np.uint8)

        if not (np.all(np.isfinite(vasc_array)) and
                np.all((vasc_array >= 0) & (vasc_array <= 255)) and
                np.all(np.equal(vasc_array, np.round(vasc_array)))):
            raise ValueError(
                f"In {stack_path}: vasc images must be safely castable to uint8")

        vasc = vasc_array.astype(np.uint8)

        # Enforce divisibility by 8 in dimensions to avoid problems in vascular segmentation
        new_pixel_size = int(np.floor(gbm.shape[0] / 8) * 8)
        gbm = gbm[0:new_pixel_size, 0:new_pixel_size, :]
        vasc = vasc[0:new_pixel_size, 0:new_pixel_size, :]

        return gbm, vasc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_tracking(self,
                     save_point=None,
                     load_point=None,
                     detection_sensitivity=None,
                     detection_backup=None
                     ):
        """Run tracking algorithm."""

        # First check if the user wants to reload tracking for some point
        lp = load_point
        if lp is not None:
            loaded = self.__class__.load_run(lp)

            # compare your current initialized subtable/stacktable vs loaded one
            self._assert_same_rows_except_tracked(
                self.subtable, loaded.subtable)

            self.__dict__.update(loaded.__dict__)

        # If the loaded point has already been fully tracked
        self._tracked = np.all(self.subtable["tracked"] == True)
        if self._tracked:
            print("All stacks have already been tracked")

        # Update the detection sensitivity:
        if detection_sensitivity is not None:
            ds = validate_detection_sensitivity(detection_sensitivity)
        else:
            ds = self.detection_sensitivity

        # Validate backup detection
        if detection_backup is not None:
            detection_backup = validate_detection_sensitivity(detection_backup)

        # -------------------------------------------
        # ----------- Begin computations ------------
        # -------------------------------------------

        # Order channels
        for key, ch in self.channel_roles.items():
            if ch == "gbm":
                gbm_channel = key
            if ch == "vasc":
                vasc_channel = key

        # Load networks
        gbm_net, tme_net, seg_net = load_trained_networks()
        blocksize = 61

        # indices of rows that are NOT tracked (preserves original row labels)
        untracked_idx = self.subtable.index[~self.subtable["tracked"].astype(
            bool)].to_list()
        total_untracked = len(untracked_idx)

        print("Number of untracked stacks:", total_untracked)

        # -------------------------------------------
        # ----------- Run tracking loop -------------
        # -------------------------------------------
        for row_i in tqdm(untracked_idx, desc="Tracking and classifying cells", unit="stack"):

            stack_path = Path(self.subtable.loc[row_i, "file_path"])
            dt = self.subtable.loc[row_i, "delta_t"]
            data = np.load(stack_path, allow_pickle=True)

            # Map channels
            channel_data = {
                "blue": data["Bstack"],
                "green": data["Tstack"],
                "red": data["Vstack"],
            }

            # Make sure conditons are met for image tracking and classifcation
            gbm_array = channel_data[gbm_channel]
            vasc_array = channel_data[vasc_channel]

            gbm, vasc = self._prepare_gbm_vasc_arrays(
                gbm_array, vasc_array, stack_path)

            # Track stack
            tracked_stack = build_tracks_and_vascularity(
                gbm=gbm,
                vasc=vasc,
                gbm_net=gbm_net,
                tme_net=tme_net,
                seg_net=seg_net,
                blocksize=blocksize,
                detection_sensitivity=ds,
                i=row_i,
                dt=dt
            )

            # Retry tracking if no detection and detection_backup was provided
            if tracked_stack is None and detection_backup is not None:
                print(
                    f"Warning, Due to no detections found in {row_i}: exp {self.subtable.loc[row_i, "exp"]}, roi {self.subtable.loc[row_i, "roi"]}. Retrying with DS = {detection_backup}")
                tracked_stack = build_tracks_and_vascularity(
                    gbm=gbm,
                    vasc=vasc,
                    gbm_net=gbm_net,
                    tme_net=tme_net,
                    seg_net=seg_net,
                    blocksize=blocksize,
                    detection_sensitivity=detection_backup,
                    i=row_i,
                    dt=dt
                )

            # Drop stack if tracking fails to detect
            if tracked_stack is None:
                print(
                    f"Warning, Due to no detections found. dropped row {row_i}: exp {self.subtable.loc[row_i, "exp"]}, roi {self.subtable.loc[row_i, "roi"]}")
                self.subtable = self.subtable.drop(index=row_i)
                # Need to check all tracked if this was the last stack
                np.all(self.subtable["tracked"] == True)
                continue

            # Tag the tracked stack with information about the stack
            tracked_stack["frame_size"] = gbm.shape[0]
            tracked_stack["stack_index"] = row_i
            tracked_stack["delta_t"] = dt
            tracked_stack["exp"] = self.subtable.loc[row_i, "exp"]
            tracked_stack["roi"] = self.subtable.loc[row_i, "roi"]
            tracked_stack["is_treatment"] = self.subtable.loc[row_i,
                                                              "is_treatment"]

            # If patient informaiton exists as well:
            if "patient_id" in self.subtable.columns:
                tracked_stack["patient_id"] = str(
                    self.subtable.loc[row_i, "patient_id"])

            if "set" in self.subtable.columns:
                tracked_stack["set"] = int(self.subtable.loc[row_i, "set"])

            # Append the information to preexisting track
            if self.track_data is not None:
                self.track_data = pd.concat([self.track_data, tracked_stack])
            else:
                self.track_data = tracked_stack

            # Mark the stack as tracked
            self.subtable.loc[row_i, "tracked"] = True
            # Check if everything is tracked
            self._tracked = np.all(self.subtable["tracked"] == True)

            # Save current run to allow resuming later
            if save_point is not None:
                self.save_run(save_point)

    def glm_hmm(self, fcols: Optional[list[str]] = None, hmm_param: Optional[dict] = None):
        """Run GLM-HMM pipeline."""
        self._require_tracked()

        if fcols is not None:
            self.fcols = validate_fcols(fcols)

        if hmm_param is not None:
            self.hmm_param = validate_hmm_param(hmm_param)

        self._data_feat_unfilt, self.data_feat, self.pi, self.glm_models, self.A_global, self.gammas, self.loglik = hmm_pipeline(
            self.track_data,
            self.fcols,
            self.hmm_param,
        )

        self._hmm = True

    def get_loglik(self):
        """Get the log likelihood from the hmm run."""
        self._require_tracked()
        self._require_hmm()
        return self.loglik
    
    def generate_summary_statistics(
        self,
        fcol,
        patient_id=None,     # str or None
        set_id=None,            # int or None
        treatment=None,   # None => both; True/False => filter
        mode="mean",
    ):
        self._require_hmm()
        fcol = validate_fcols([fcol])[0]

        agg_map = {
            "mean": "mean",
            "median": "median",
            "sum": "sum",
            "min": "min",
            "max": "max",
            "std": "std",
            "count": "count",
        }
        if mode not in agg_map:
            raise ValueError(
                f"Unsupported mode='{mode}' use one of: {sorted(agg_map)}")

        df = self._data_feat_unfilt

        # -------------------------
        # 1) Filter (patient)
        # -------------------------
        if patient_id is not None:
            if "patient_id" not in df.columns:
                raise ValueError(
                    "patient_id was provided, but data has no 'patient_id' column")
            avail_patients = set(
                df["patient_id"].dropna().astype(str).unique())
            if str(patient_id) not in avail_patients:
                raise ValueError(
                    f"patient_id={patient_id} not found in data. Available: {sorted(avail_patients)}")

            df = df[df["patient_id"].astype(str) == str(patient_id)]
        else:
            if "patient_id" in df.columns:
                raise ValueError(
                    "data has 'patient_id' column, so you must specify patient_id")

        # -------------------------
        # 2) Filter (set)
        # -------------------------
        if set_id is not None:
            if "set" not in df.columns:
                raise ValueError(
                    "set was provided, but data has no 'set' column")
            avail_sets = set(df["set"].dropna().unique())
            if int(set_id) not in avail_sets:
                raise ValueError(
                    f"set={set_id} not found for patient_id={patient_id}. Available sets for this patient: {sorted(avail_sets)}"
                )

            df = df[df["set"] == int(set_id)]
        else:
            if "set" in df.columns:
                raise ValueError(
                    f"data has 'set' column, so you must specify set (patient_id={patient_id})")

        # -------------------------
        # 3) Filter (Treatment)
        # -------------------------
        if treatment is not None:
            if "is_treatment" not in df.columns:
                raise ValueError(
                    "is_treatment was provided, but data has no 'is_treatment' column")

            avail_flags = set(
                df["is_treatment"].dropna().astype(bool).unique())
            want = bool(treatment)
            if want not in avail_flags:
                raise ValueError(
                    f"is_treatment={want} not available for patient_id={patient_id}, set={set_id}. "
                    f"Available is_treatment values here: {sorted(avail_flags)}"
                )

            df = df[df["is_treatment"] == want]

        # -------------------------
        # 4) Compute time x roi for each exp
        # -------------------------
        exp_ids = sorted(df["exp"].dropna().astype(int).unique())
        out = {}

        for exp_id in exp_ids:
            d = df.loc[df["exp"] == exp_id, ["time", "roi", fcol]].copy()
            out[int(exp_id)] = (
                d.groupby(["time", "roi"], dropna=False)[fcol]
                .agg(agg_map[mode])
                .unstack("roi")
                .sort_index()
            )

        return out

    def generate_video_compare(self, exp, roi, output: Optional[PathLike] = None):
        """
        Generate a comparison video (old labels vs Viterbi) for a given experiment + roi.

        Requires:
        - tracked data (self.track_data)
        - HMM

        output:
        - None => auto folder under ./gliotrace_videos/exp_{exp}/roi_{roi}
        - directory => created if missing
        - file path => parent created if missing
        """
        self._require_tracked()
        self._require_hmm()

        exp, roi = self._validate_exp_roi(exp, roi)
        out_path = self._coerce_output_path(output, exp, roi)

        result = generate_video_compare(
            self.track_data,
            self.subtable,
            self.data_feat,
            exp,
            roi,
            self.channel_roles,
            out_path,
        )

        # Store saved path
        self._generate_video_visualizations.append(
            result if result is not None else out_path
        )
        return result

    def generate_video(self, exp, roi, output: Optional[PathLike] = None):
        """
        Generate a video for a given experiment + roi.

        output:
          - None => auto folder under ./gliotrace_videos/exp_{exp}/roi_{roi}
          - directory => created if missing
          - file path => parent created if missing
        """
        self._require_tracked()

        exp, roi = self._validate_exp_roi(exp, roi)
        out_path = self._coerce_output_path(output, exp, roi)

        result = generate_video(
            self.subtable,
            self.track_data,
            exp,
            roi,
            self.channel_roles,
            out_path,
        )

        # Store saved path.
        self._generate_video_visualizations.append(
            result if result is not None else out_path)

    def print_exp_roi(self):
        print(self.subtable[["exp", "roi"]])

    def compare_hmm_and_cnn_class(self, threshold=0.1):

        self._require_tracked()
        if not getattr(self, "_hmm", False):
            raise RuntimeError(
                "Run glm_hmm() first so data_feat and gammas are available.")

        softmax_cols = SOFTMAX_COLUMNS
        K = len(softmax_cols)

        data_feat = self.data_feat.copy()
        gammas_long = self.gammas.copy()

        # ---- sanity checks ----
        required_df_cols = {"exp", "roi", "cellID", "time", "state_label"}
        missing = required_df_cols - set(data_feat.columns)
        if missing:
            raise ValueError(
                f"data_feat is missing required columns: {sorted(missing)}")

        required_g_cols = {"exp", "roi", "cellID", "t"}
        missing_g = required_g_cols - set(gammas_long.columns)
        if missing_g:
            raise ValueError(
                f"gammas_long is missing required columns: {sorted(missing_g)}")

        missing_softmax_df = [
            c for c in softmax_cols if c not in data_feat.columns]
        if missing_softmax_df:
            raise ValueError(
                f"data_feat is missing softmax columns: {missing_softmax_df}")

        missing_softmax_g = [
            c for c in softmax_cols if c not in gammas_long.columns]
        if missing_softmax_g:
            raise ValueError(
                f"gammas_long is missing softmax columns: {missing_softmax_g}")

        # ---- IMPORTANT: build per-track time index 't' BEFORE filtering ----
        data_feat = data_feat.sort_values(
            ["exp", "roi", "cellID", "time"]).copy()
        data_feat["t"] = data_feat.groupby(["exp", "roi", "cellID"]).cumcount()

        # ---- convert state_label to softmax label names (NO permutation) ----
        # Accepts:
        #   - 1..K (your case)
        #   - 0..K-1
        #   - already a string label in softmax_cols
        def _state_to_label(x):
            if pd.isna(x):
                return x
            if isinstance(x, str):
                if x in softmax_cols:
                    return x
                raise ValueError(
                    f"state_label string {x!r} not in SOFTMAX_COLUMNS.")
            xi = int(x)
            if 1 <= xi <= K:          # 1-based
                return softmax_cols[xi - 1]
            if 0 <= xi < K:           # 0-based
                return softmax_cols[xi]
            raise ValueError(
                f"state_label {xi} out of range; expected 1..{K} or 0..{K-1}.")

        data_feat["state_label"] = data_feat["state_label"].map(
            _state_to_label)

        # ---- build mask: max CNN softmax prob > threshold ----
        maxprob = data_feat[softmax_cols].max(axis=1)
        data_sel = data_feat.loc[maxprob > threshold].copy()
        if data_sel.empty:
            print("No rows passed the CNN threshold.")
            return pd.DataFrame()

        # ---- merge selected data with gammas ----
        merged = data_sel.merge(
            gammas_long[["exp", "roi", "cellID", "t"] + softmax_cols],
            on=["exp", "roi", "cellID", "t"],
            how="inner",
            validate="one_to_one",
            suffixes=("_cnn", "_gamma"),
        )

        # ---- gamma class from GAMMA columns ----
        gamma_cols = [f"{c}_gamma" for c in softmax_cols]
        merged["gamma_class"] = (
            merged[gamma_cols]
            .idxmax(axis=1)
            .str.replace(r"_gamma$", "", regex=True)
        )

        # ---- contingency table: state_label vs gamma_class ----
        ct = (
            merged
            .groupby(["state_label", "gamma_class"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=softmax_cols, columns=softmax_cols, fill_value=0)
        )

        return ct

    # ------------------------------------------------------------------
    # Pretty printing / debug
    # ------------------------------------------------------------------

    def _print_configuration(self):
        print("\n=== Gliotrace configuration ===")
        st = self.subtable
        print(f"Number of stacks        : {len(st)}")

        def _show(xs, n=10):
            xs = list(xs)
            if len(xs) <= n:
                return xs
            return xs[:n] + [f"... (n={len(xs)})"]

        print("Scope                   :")

        has_patient = "patient_id" in st.columns
        has_set = "set" in st.columns
        has_exp = "exp" in st.columns

        if has_patient and has_set and has_exp:
            # patient -> set -> exps
            grp = (
                st.dropna(subset=["patient_id", "set", "exp"])
                .groupby(["patient_id", "set"])["exp"]
                .apply(lambda s: sorted(s.astype(int).unique().tolist()))
            )
            for pid in _show(sorted(grp.index.get_level_values(0).unique()), n=20):
                if isinstance(pid, str) and pid.startswith("..."):
                    print(f"  {pid}")
                    break
                print(f"  Patient {pid}:")
                sub = grp.loc[pid]  # indexed by set
                for s in _show(sorted(sub.index.unique().tolist()), n=20):
                    if isinstance(s, str) and s.startswith("..."):
                        print(f"    {s}")
                        break
                    print(f"    Set {s}: exps {_show(sub.loc[s], n=12)}")

        elif has_patient and has_exp:
            # patient -> exps
            grp = (
                st.dropna(subset=["patient_id", "exp"])
                .groupby("patient_id")["exp"]
                .apply(lambda s: sorted(s.astype(int).unique().tolist()))
            )
            for pid in _show(sorted(grp.index.unique().tolist()), n=20):
                if isinstance(pid, str) and pid.startswith("..."):
                    print(f"  {pid}")
                    break
                print(f"  Patient {pid}: exps {_show(grp.loc[pid], n=12)}")

        elif has_set and has_exp:
            # set -> exps
            grp = (
                st.dropna(subset=["set", "exp"])
                .groupby("set")["exp"]
                .apply(lambda s: sorted(s.astype(int).unique().tolist()))
            )
            for s in _show(sorted(grp.index.unique().tolist()), n=30):
                if isinstance(s, str) and s.startswith("..."):
                    print(f"  {s}")
                    break
                print(f"  Set {s}: exps {_show(grp.loc[s], n=12)}")

        elif has_exp:
            exps = sorted(st["exp"].dropna().astype(int).unique().tolist())
            print(f"  Exps                  : {_show(exps, n=20)}")
        else:
            print("  (no 'exp' column found)")

        # ----- core config -----
        print(f"Detection sensitivity   : {self.detection_sensitivity}")
        print(f"Channel roles           : {self.channel_roles}")
        print(f"Feature columns         : {self.fcols}")
        print(f"HMM parameters          : {self.hmm_param}")

        # ----- controls / treatment -----
        if hasattr(self, "control"):
            print(f"Controls                : {self.control}")

        if getattr(self, "treatment", None) is not None:
            t_name, t_dose = self.treatment
            if t_dose is None:
                print(f"Treatment               : {t_name}")
            else:
                print(f"Treatment               : {t_name} (dose={t_dose})")

        if "is_treatment" in st.columns:
            n_treat = int((st["is_treatment"] == True).sum())
            n_ctrl = int((st["is_treatment"] == False).sum())
            print(f"Stacks (control/treat)  : {n_ctrl} / {n_treat}")

        print("================================\n")

    def save_run(self, run_dir: str | Path) -> Path:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "tables").mkdir(exist_ok=True)

        manifest = {
            "format": "gliotrace-run-v1",
            "tracked": bool(getattr(self, "_tracked", False)),
            "hmm": bool(getattr(self, "_hmm", False)),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        cfg = {
            "detection_sensitivity": self.detection_sensitivity,
            "channel_roles": self.channel_roles,
            "fcols": self.fcols,
            "hmm_param": self.hmm_param,
            "control": self.control,
            "treatment": getattr(self, "treatment", None),
            "verbose": getattr(self, "verbose", False),
        }
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

        # --- tables (pickle handles object columns like segmented_stack) ---
        self.subtable.to_pickle(run_dir / "tables" / "subtable.pkl")

        if getattr(self, "track_data", None) is not None:
            self.track_data.to_pickle(run_dir / "tables" / "track_data.pkl")

        if getattr(self, "_hmm", False):
            if getattr(self, "data_feat", None) is not None:
                self.data_feat.to_pickle(run_dir / "tables" / "data_feat.pkl")
            if getattr(self, "gammas", None) is not None:
                self.gammas.to_pickle(run_dir / "tables" / "gammas.pkl")

            np.savez_compressed(
                run_dir / "arrays.npz",
                pi=np.asarray(getattr(self, "pi", [])),
                A_global=np.asarray(getattr(self, "A_global", [])),
            )

            if joblib is None:
                raise RuntimeError(
                    "joblib is required to save glm_models (pip install joblib).")
            joblib.dump(getattr(self, "glm_models", None),
                        run_dir / "glm_models.joblib")

        return run_dir

    @classmethod
    def load_run(cls, run_dir: str | Path) -> "Gliotrace":
        run_dir = Path(run_dir)

        manifest = json.loads((run_dir / "manifest.json").read_text())
        if manifest.get("format") != "gliotrace-run-v1":
            raise ValueError(f"Unknown run format: {manifest.get('format')}")

        cfg = json.loads((run_dir / "config.json").read_text())

        self = cls.__new__(cls)

        # restore config fields
        self.verbose = cfg.get("verbose", False)
        self.detection_sensitivity = cfg["detection_sensitivity"]
        self.channel_roles = cfg["channel_roles"]
        self.fcols = cfg["fcols"]
        self.hmm_param = cfg["hmm_param"]
        self.control = cfg["control"]
        self.treatment = cfg.get("treatment", None)

        # internal flags
        self._tracked = bool(manifest.get("tracked", False))
        self._hmm = bool(manifest.get("hmm", False))
        self._generate_video_visualizations = []

        # load tables (pickle)
        self.subtable = pd.read_pickle(run_dir / "tables" / "subtable.pkl")

        track_path = run_dir / "tables" / "track_data.pkl"
        if track_path.exists():
            self.track_data = pd.read_pickle(track_path)
        else:
            self.track_data = None

        if self._hmm:
            data_feat_path = run_dir / "tables" / "data_feat.pkl"
            gammas_path = run_dir / "tables" / "gammas.pkl"
            self.data_feat = pd.read_pickle(
                data_feat_path) if data_feat_path.exists() else None
            self.gammas = pd.read_pickle(
                gammas_path) if gammas_path.exists() else None

            arrays_path = run_dir / "arrays.npz"
            if arrays_path.exists():
                arrays = np.load(arrays_path, allow_pickle=False)
                self.pi = arrays["pi"]
                self.A_global = arrays["A_global"]
            else:
                self.pi = None
                self.A_global = None

            if joblib is None:
                raise RuntimeError(
                    "joblib is required to load glm_models (pip install joblib).")
            glm_path = run_dir / "glm_models.joblib"
            self.glm_models = joblib.load(
                glm_path) if glm_path.exists() else None

        return self
