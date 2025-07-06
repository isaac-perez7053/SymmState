import numpy as np
from typing import List, Tuple, Any, Dict
from collections import OrderedDict, defaultdict
from itertools import groupby
import warnings

from pymatgen.symmetry.bandstructure import HighSymmKpath
from symmstate.abinit import AbinitFile
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import plotly.graph_objects as go
from symmstate.utils import SymmAdaptedBasis


class SymmAdaptStruct:
    """ """

    def __init__(self, abi_file: AbinitFile, abi_eigs: str):
        # TODO: I just need all irreps, then run the smodes_calc to find the
        self.abi_file = abi_file
        self.symmeigs = SymmAdaptedBasis.all_modes(abi_file)
        self.abi_eigs = self._parse_eigs(abi_eigs)
        # Extract eigenmodes from Abinit
        self._parse_eigs(abi_eigs)

    def _parse_eigs(
        self, abi_eigs: str
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, int, float, float]]]:
        """
        Parse Abinit eigenvalue file and extract all q-points and eigenmodes.

        Returns:
            - qpoints: np.ndarray of shape (n_qpoints, 3) from the "Grid q points" section
            - eigenmodes: list of tuples (q_vector_from_mode, eigendisplacement, mode_number, energy)
        """

        try:
            with open(abi_eigs, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"File not found: {abi_eigs}")
        except IOError as e:
            print(f"Error opening file {abi_eigs}: {e}")

        qpoints = []
        eigenmodes = []
        idx = 0

        # Extract full q-point grid from "Grid q points"
        while idx < len(lines):
            line = lines[idx]
            if "Grid q points" in line:
                n_qpoints = int(line.split(":")[1].strip())
                for i in range(n_qpoints):
                    qline = lines[idx + 1 + i]
                    coords = qline.split(")", 1)[1].strip().split()
                    qpoints.append([float(x) for x in coords])
                qpoints = np.array(qpoints)
                idx += n_qpoints + 1
                break
            idx += 1

        # Extract eigenmodes
        current_q_vector = None
        while idx < len(lines):
            line = lines[idx].strip()

            # Detect and store q-vector (per mode block)
            if "Phonon wavevector (reduced coordinates)" in line:
                current_q_vector = np.array(
                    [float(x) for x in line.split(":")[1].strip().split()]
                )
                idx += 1
                continue

            elif "Phonon at Gamma, with non-analyticity" in line:
                current_q_vector = np.array(
                    [1.0, 0.0, 0.0]
                )  # Gamma point in the next cell
                idx += 2  # skip the line about non-analyticity direction
                continue

            # Detect and store frequencies (in cm-1)
            elif "Phonon frequencies in cm-1" in line:
                pending_frequencies = []
                idx += 1
                while idx < len(lines):
                    freq_line = lines[idx].strip()
                    if not (freq_line.startswith("-") or freq_line.startswith(";")):
                        break
                    # Remove leading marker and split values
                    freqs = [float(val) for val in freq_line[1:].strip().split()]
                    pending_frequencies.extend(freqs)
                    idx += 1
                continue

            # Look for the mdoe
            if line.startswith("Mode number"):
                parts = line.split()
                mode_number = int(parts[2])
                energy = float(parts[4])
                frequency_cm1 = (
                    pending_frequencies[mode_number - 1]
                    if len(pending_frequencies) >= mode_number
                    else None
                )
                real_parts = []
                imag_parts = []
                idx += 1

                # Skip optional warning lines if necessary
                while idx < len(lines) and (
                    "Attention" in lines[idx]
                    or "unstable" in lines[idx]
                    or lines[idx].strip() == ""
                ):
                    idx += 1

                # Read eigendisplacement blocks
                while idx + 1 < len(lines):
                    real_line = lines[idx].strip()
                    imag_line = lines[idx + 1].strip()

                    if not (real_line.startswith("-") or real_line.startswith(";")):
                        break

                    real_values = real_line[1:].strip().split()
                    if len(real_values) < 4:
                        break  # not a valid vector line

                    try:
                        real_vec = [float(x) for x in real_values[1:]]
                        imag_vec = [float(x) for x in imag_line[1:].strip().split()]
                    except ValueError:
                        break

                    real_parts.append(real_vec)
                    imag_parts.append(imag_vec)
                    idx += 2

                real_arr = np.array(real_parts)
                imag_arr = np.array(imag_parts)
                eigendisplacement = real_arr + 1j * imag_arr

                eigenmodes.append(
                    (
                        current_q_vector,
                        eigendisplacement,
                        mode_number,
                        energy,
                        frequency_cm1,
                    )
                )
            else:
                idx += 1

        self.qpoints = qpoints
        self.eigenmodes = eigenmodes

    def get_high_symmetry_qpoint_labels(self):
        """
        Returns a dictionary mapping q-points (as tuples) to their labels
        """
        kpath = HighSymmKpath(self.abi_file)
        label_map = {}

        for label, frac_coords in kpath.kpath["kpoints"].items():
            key = tuple(np.round(frac_coords, 6))
            label_map[key] = label.replace("\Gamma", "GM")

        label_map = {tuple(np.round(k, 6)): v for k, v in label_map.items()}
        return label_map

    def package_phonons(self):
        """
        Package all phonons along high-symmetry paths, tagging those at special q-points with labels.
        """
        phonons = []
        irrep = None
        degeneracy = None

        # Use pymatgen to get high-symmetry q-point labels
        q_label_map = self.get_high_symmetry_qpoint_labels(self.abi_file)

        special_q_labels = []
        special_q_paths = []

        special_q_map = OrderedDict()

        for current_q_vector, eigendisplacement, _, _, frequency_cm1 in self.eigenmodes:
            q_vec_rounded = tuple(np.round(current_q_vector, 6))
            label = q_label_map.get(q_vec_rounded)

            phonon = (
                current_q_vector,
                eigendisplacement,
                frequency_cm1,
                irrep,
                degeneracy,
            )
            phonons.append(phonon)

            if label and q_vec_rounded not in special_q_map:
                special_q_map[q_vec_rounded] = (label, current_q_vector.tolist())

        # Extract clean outputs
        special_q_labels = [v[0] for v in special_q_map.values()]
        special_q_paths = [v[1] for v in special_q_map.values()]

        self.phonons = phonons
        self.special = (special_q_labels, special_q_paths)

    def build_full_label_map(self) -> defaultdict[Any, set]:
        """
        Returns a defaultdict(set) mapping every symmetry-equivalent q-vector
        in [0,1)^3 to its set of high-symmetry labels.
        """
        # get the “standard” high-symmetry dictionary from pymatgen
        raw_map = self.get_high_symmetry_qpoint_labels(
            self.abi_file.get_primitive_structure()
        )

        # canonicalize into a defaultdict(set)
        label_map = defaultdict(set)
        for q, lab in raw_map.items():
            q = np.array(q) % 1
            key = tuple(np.round(q, 4))
            label_map[key].add(lab)

        # get the fractional rotation matrices
        sga = SpacegroupAnalyzer(self.abi_file)
        ops = sga.get_symmetry_operations(cartesian=False)

        # apply every rotation to every base point
        full_map = defaultdict(set)
        for q0, labs in label_map.items():
            q0 = np.array(q0)
            for op in ops:
                q_sym = op.rotation_matrix.dot(q0)
                q_sym = tuple(np.round(q_sym % 1, 4))
                full_map[q_sym].update(labs)

        return full_map

    def detect_path(self) -> List:
        """
        Detect high-symmetry path labels from q-points in a phonon band structure.

        Args:
            phonons: A tuple where the first element is the q-point array.
            abi_file: The AbinitStructure object with method `.get_primitive_structure()`.

        Returns:
            A list of high-symmetry labels (e.g., ["Γ", "X", "W", ...]).
        """
        symm_path = []
        symm_qvec = []
        label_map = self.build_full_label_map()

        for qvec, labels in label_map.items():
            qstr = f"({qvec[0]:.3f}, {qvec[1]:.3f}, {qvec[2]:.3f})"
            label_str = ", ".join(sorted(labels))
            print(f"{qstr} → {label_str}")

        for phonon in self.phonons:
            qpoint = np.round(np.array(phonon[0]) % 1, 4)
            found_labels = label_map.get(tuple(qpoint), set())
            if found_labels:
                symm_path.append(sorted(found_labels)[0])  # Pick one label if many
                symm_qvec.append(tuple(qpoint))
            else:
                symm_path.append("")
        return list(
            zip(
                [key for key, _ in groupby(symm_path) if key],
                [np.array(key) for key, _ in groupby(symm_qvec)],
            )
        )

    def package_phonon_bands(self) -> Tuple[
        # qpath_segs: list of legs, each leg is a list of 3-vectors
        List[List[float]],
        # (band_segs, disp_segs):
        #   band_segs: list of legs -> list of modes → list of freqs
        #   disp_segs: list of legs -> list of modes → list of displacements
        Tuple[
            List[List[List[float]]],
            List[List[np.ndarray]],
        ],
        # (hs_freqs, hs_disps):
        #   hs_freqs: list of nodes -> list of freqs
        #   hs_disps: list of nodes -> list of disps
        Tuple[
            List[List[float]],
            List[List[np.ndarray]],
        ],
    ]:
        Natoms = self.abi_file.vars["natom"]
        Nmode = 3 * Natoms
        Nq = len() // Nmode

        # extract one q per block
        qpath = np.array([self.phonons[i * Nmode][0] for i in range(Nq)])
        freqs = np.array([p[2] for p in self.phonons]).reshape(Nq, Nmode)
        disps = np.array([p[1] for p in self.phonons]).reshape(Nq, Nmode, Natoms, 3)

        # build HS label→indices
        symm_path = self.detect_path()
        symm_lookup = {tuple(np.round(q % 1, 4)): lbl for lbl, q in symm_path}
        label_to_indices = defaultdict(list)
        for iq, q in enumerate(qpath):
            lab = symm_lookup.get(tuple(np.round(q % 1, 4)))
            if lab:
                label_to_indices[lab].append(iq)

        # pick the first occurrence of each label, in order
        idx_path = [label_to_indices[lbl][0] for lbl, _ in symm_path]

        # containers for each segment
        qpath_segs = []
        band_segs = []
        disp_segs = []
        hs_freqs = []
        hs_disps = []

        # record the very first HS point
        hs_freqs.append(freqs[idx_path[0]].tolist())
        hs_disps.append(disps[idx_path[0]].tolist())

        # now loop over legs
        for i_start, i_end in zip(idx_path, idx_path[1:]):
            step = 1 if i_end > i_start else -1
            seg_idx = np.arange(i_start, i_end + step, step)

            # slice out this leg’s qpoints
            qpath_segs.append(qpath[seg_idx].tolist())

            # slice bands and disps mode‐by‐mode
            band_segs.append([freqs[seg_idx, m].tolist() for m in range(Nmode)])
            disp_segs.append([disps[seg_idx, m] for m in range(Nmode)])

            # record the high‐symm point at the end
            hs_freqs.append(freqs[i_end].tolist())
            hs_disps.append(disps[i_end].tolist())

        return qpath_segs, (band_segs, disp_segs), (hs_freqs, hs_disps)

    def find_all_irreps(self, qvec):
        """ """
        # Logic to match qvec with high symmetry label (i.e. GM or X or DT)
        pass

    def find_all_phonon_labels(
        self,
        qpath_segs: List[List[List[float]]],
        disp_segs: List[List[np.ndarray]],
    ) -> List[List[Dict[str, Any]]]:
        """
        Projects phonon eigendisplacements onto symmetry-adapted subspaces and assigns symmetry labels.

        Parameters:
            abi_file: AbinitFile with symmetry information.
            qpath_segs: List of q-vector segments (one per path leg).
            disp_segs: List of displacement segments (modes × q-points).
            symm_path: Ordered list of (label, qvec) for high-symmetry points.

        Returns:
            List of lists of phonon label data per segment.
            Each element is a list of dicts:
                {
                    "qvec": List[float],
                    "mode_index": int,
                    "sym_label": str,
                    "is_high_symmetry": bool,
                    "high_sym_label": Optional[str],
                }
        """
        phonon_labels = []

        # Extract just the list of high-symmetry q-vectors for easy comparison
        hs_qpoints = {
            tuple(np.round(np.array(q) % 1, 5)): label
            for label, q in self.detect_path()
        }

        for seg_index, (qseg, disp_modes) in enumerate(zip(qpath_segs, disp_segs)):
            seg_label_data = []

            for iq, qvec in enumerate(qseg):
                rounded_qvec = tuple(np.round(np.array(qvec) % 1, 5))
                is_hs_point = rounded_qvec in hs_qpoints
                hs_label = hs_qpoints.get(rounded_qvec, None)

                # Get irreps for this q-point
                irreps, degen = self.find_all_irreps(qvec)

                for imode, disp in enumerate(disp_modes):
                    eigendisplacement = disp[
                        iq
                    ]  # (Natom, 3) array at this qvec for this mode

                    # Project onto symmetry-adapted subspaces and get label
                    phonon_sym_label = self.find_phonon_label(eigendisplacement, irreps)

                    seg_label_data.append(
                        {
                            "qvec": qvec,
                            "mode_index": imode,
                            "sym_label": phonon_sym_label,
                            "is_high_symmetry": is_hs_point,
                            "high_sym_label": hs_label,
                            "degen": degen,
                        }
                    )

            phonon_labels.append(seg_label_data)

        return phonon_labels, self.build_irrep_lookup(phonon_labels)

    def find_phonon_label(
        self,
        eigendisplacement: np.ndarray,
        irreps: Dict[str, List[np.ndarray]],
        threshold: float = 0.8,
    ) -> str:
        """
        Project a phonon eigendisplacement onto each irrep subspace (orthonormalized),
        and return the best-matching irrep label. Warn if overlap is below threshold.

        Parameters:
            eigendisplacement: np.ndarray of shape (Natom, 3) or flattened (3*Natom,)
            irreps: Dict[str, List[np.ndarray]] of (not necessarily orthonormal) irrep bases
            threshold: minimum squared norm (0-1) required for a confident match

        Returns:
            best_irrep: str label of best-matching irrep
        """
        if eigendisplacement.ndim == 2:
            eigendisplacement = eigendisplacement.flatten()

        eigendisplacement = eigendisplacement / np.linalg.norm(eigendisplacement)

        best_irrep = None
        max_overlap = -1.0

        for label, basis in irreps.items():
            B = np.stack(basis, axis=1)  # shape: (3*Natom, num_vectors)
            Q, _ = np.linalg.qr(B)  # Orthonormalize basis vectors just in case

            # Orthogonal projection
            projection = Q @ (Q.T @ eigendisplacement)
            overlap = np.linalg.norm(projection) ** 2

            if overlap > max_overlap:
                max_overlap = overlap
                best_irrep = label

        if max_overlap < threshold:
            warnings.warn(
                f"Low projection confidence: {100*max_overlap:.1f}% in '{best_irrep}' subspace. "
                f"This eigendisplacement may not correspond cleanly to any single irrep.",
                category=UserWarning,
            )

        return best_irrep, max_overlap

    def build_irrep_lookup(
        self, phonon_labels: List[List[Dict[str, Any]]]
    ) -> Dict[Tuple[str, int], Tuple[str, int]]:
        """
        Builds a lookup table from (high_sym_label, mode_index) to (irrep_label, degeneracy)
        by inspecting only high-symmetry points.
        """
        irrep_modes = defaultdict(list)

        for seg in phonon_labels:
            for entry in seg:
                if entry["is_high_symmetry"] and entry["high_sym_label"] is not None:
                    key = (entry["high_sym_label"], entry["mode_index"])
                    irrep_modes[key].append(entry["sym_label"])

        # Determine degeneracies and collapse to most common label
        irrep_lookup = {}
        for key, labels in irrep_modes.items():
            label_counts = defaultdict(int)
            for label in labels:
                label_counts[label] += 1
            best_label = max(label_counts, key=label_counts.get)
            degeneracy = label_counts[best_label]
            irrep_lookup[key] = (best_label, degeneracy)

        return irrep_lookup

    def plot_phonon_band_structure(self, output_name="phonon_dispersion_curve.html"):
        """
        Create an interactive plot of the phonon band structure including
        symmetry labels, degeneracy, and decomposition of the phonons.
        """
        # Parse and package
        symm_path = self.detect_path()
        qpath_segs, (band_segs, _), (hs_freqs, _) = self.package_phonon_bands()

        # Ordered HS labels
        hs_labels = [lbl for lbl, _ in symm_path]

        # Build x-axis for each segment: each goes from i to i+1
        x_segs = []
        for i, qseg in enumerate(qpath_segs):
            npts = len(qseg)
            x_segs.append(np.linspace(i, i + 1, npts))

        fig = go.Figure()

        # phonon lines for each leg
        for seg_idx, (xseg, modes) in enumerate(zip(x_segs, band_segs)):
            for mode_idx, freqs in enumerate(modes):
                fig.add_trace(
                    go.Scatter(
                        x=xseg,
                        y=freqs,
                        mode="lines",
                        line=dict(width=1),
                        name=f"Mode {mode_idx+1}",
                        showlegend=(seg_idx == 0),
                    )
                )

        # vertical dashed lines at each node
        for i in range(len(hs_labels)):
            fig.add_vline(x=i, line=dict(color="black", dash="dash", width=1))

        # configure the x-axis to show exactly one set of HS labels
        fig.update_xaxes(
            title="Wave Vector",
            tickmode="array",
            tickvals=list(range(len(hs_labels))),
            ticktext=hs_labels,
            showgrid=False,
        )

        # high-symmetry markers with irreps & degeneracies
        for node_idx, (lbl, _) in enumerate(symm_path):
            x = node_idx  # integer position
            for mode_idx, freq in enumerate(hs_freqs[node_idx]):
                irrep = self.find_irrep(lbl, mode_idx, freq)
                degen = self.find_degeneracy(lbl, mode_idx, freq)
                fig.add_trace(
                    go.Scatter(
                        x=[x],
                        y=[freq],
                        mode="markers",
                        marker=dict(symbol="diamond", size=8, color="black"),
                        showlegend=False,
                        customdata=[[irrep, degen]],
                        hovertemplate=(
                            "Mode %{customdata[1]}<br>"
                            "f = %{y:.3f} THz<br>"
                            "Irrep: %{customdata[0]}<br>"
                            "Degeneracy: %{customdata[1]}<extra></extra>"
                        ),
                    )
                )

        # layout tweaks
        fig.update_layout(
            title="Phonon Dispersion",
            xaxis=dict(
                title="High‐symmetry path",
                tickmode="array",
                tickvals=list(range(len(hs_labels))),
                ticktext=hs_labels,
                showgrid=False,
            ),
            yaxis=dict(title="Frequency (THz)"),
        )

        #  write out
        fig.write_html(output_name)
