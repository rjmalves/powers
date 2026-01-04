"""CPU profiling collector using perf and FlameGraph."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import ProfilingConfig
from ..schemas import CollectorResult
from .base import Collector

PERF_DATA = "perf.data"
PERF_SCRIPT = "perf_script.txt"
FOLDED = "perf.folded"
FLAMEGRAPH = "flamegraph.svg"
HOTSPOTS = "hotspots.json"


def _which_tool(candidate: Optional[Path], fallback: str) -> Optional[Path]:
    if candidate:
        if candidate.exists():
            return candidate
        return None
    resolved = shutil.which(fallback)
    return Path(resolved) if resolved else None


def _run_command(
    command: List[str],
    cwd: Path,
    timeout: int = 300,
    input_text: Optional[str] = None,
) -> Tuple[int, str, str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        input=input_text,
    )
    return result.returncode, result.stdout, result.stderr


def _parse_perf_report(stdout: str, limit: int = 20) -> List[Dict[str, str]]:
    """Parse perf report --stdio output for top hotspots.

    Expected format (with command and shared object):
      21.21%     0.00%  powers   [unknown]    [.] symbol_name
    
    Or simpler format:
      17.17%     0.00%  [.] symbol_name      -      -

    Includes [unknown] symbols with descriptive placeholders to show
    missing symbol resolution (helps users identify profiling gaps).
    """
    hotspots: List[Dict[str, str]] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        parts = stripped.split()
        if len(parts) < 4 or "%" not in parts[0]:
            continue
        try:
            percent_token = parts[0].lstrip("+*-#")
            percent_val = float(percent_token.replace("%", ""))
        except ValueError:
            continue

        # Find the symbol name after [.] or [k] marker
        # The marker can be at different positions:
        #   Format 1: percent1 percent2 [.] symbol ...
        #   Format 2: percent1 percent2 cmd shared_obj [.] symbol ...
        symbol: str = ""
        marker_idx = -1
        
        # Find the marker index
        for i, part in enumerate(parts):
            if part in ["[.]", "[k]", "[H]", "[.],"]:
                marker_idx = i
                break
        
        if marker_idx >= 0 and marker_idx + 1 < len(parts):
            # Extract everything after the marker until we hit "-" or end
            symbol_parts = []
            for i in range(marker_idx + 1, len(parts)):
                if parts[i] == "-":
                    break
                symbol_parts.append(parts[i])
            if symbol_parts:
                symbol = " ".join(symbol_parts)

        # Skip truly empty symbols
        if not symbol or symbol.strip() == "":
            continue

        # Keep [unknown] symbols but make them identifiable
        # Convert hex addresses to [unknown:<addr>] format
        if symbol.startswith("0x") or symbol.replace("0", "").replace("x", "") == "":
            # Preserve hex address for debugging
            symbol = f"[unknown:{symbol}]"
        elif symbol == "unknown" or symbol == "[unknown]":
            symbol = "[unknown]"
        elif symbol.isdigit() or (
            symbol.startswith("0")
            and all(c in "0123456789abcdefABCDEFx" for c in symbol)
        ):
            # Numeric-only symbols (likely addresses)
            symbol = f"[unknown:{symbol}]"

        entry = {
            "symbol": symbol,
            "percent": percent_val,
            "raw_line": stripped,
        }
        hotspots.append(entry)
        if len(hotspots) >= limit:
            break
    return hotspots


class CPUCollector(Collector):
    """Collector that wraps perf record/script/report and FlameGraph generation."""

    name = "cpu"

    def collect(
        self,
        *,
        binary: Path,
        args: List[str],
        config: ProfilingConfig,
        run_dir: Path,
    ) -> CollectorResult:
        start = time.perf_counter()
        collector_dir = run_dir / self.name
        collector_dir.mkdir(parents=True, exist_ok=True)

        errors: List[str] = []
        warnings: List[str] = []
        data: Dict[str, object] = {}
        raw_files: List[str] = []

        perf_path = _which_tool(config.perf_path, "perf")
        flamegraph_dir = config.flamegraph_path

        if perf_path is None:
            errors.append(
                "perf not found. Install linux-tools or set tools.perf."
            )
            duration = time.perf_counter() - start
            return CollectorResult(
                collector_name=self.name,
                success=False,
                duration_seconds=duration,
                data=data,
                errors=errors,
                warnings=warnings,
            )

        perf_data_path = collector_dir / PERF_DATA
        perf_script_path = collector_dir / PERF_SCRIPT
        folded_path = collector_dir / FOLDED
        flamegraph_path = collector_dir / FLAMEGRAPH

        perf_record_cmd = [
            str(perf_path),
            "record",
            "-F",
            str(config.perf_frequency),
            "--call-graph",
            "fp",  # Frame pointer unwinding (reliable, works without libdw)
            "--buildid-all",  # Record build IDs for symbol resolution
            "--output",
            str(perf_data_path),
            "--",
            str(binary),
            *args,
        ]

        rc, out, err = _run_command(perf_record_cmd, cwd=config.repo_root)
        raw_files.extend([str(perf_data_path)])
        data["perf_record_stdout"] = out
        data["perf_record_stderr"] = err
        data["perf_record_exit_code"] = rc
        if rc != 0:
            errors.append(f"perf record failed with exit code {rc}")

        if rc == 0:
            script_cmd = [str(perf_path), "script", "-i", str(perf_data_path)]
            rc_script, out_script, err_script = _run_command(
                script_cmd, cwd=collector_dir
            )
            perf_script_path.write_text(out_script)
            raw_files.append(str(perf_script_path))
            if rc_script != 0:
                errors.append(f"perf script failed with exit code {rc_script}")
            else:
                data["perf_script_lines"] = min(
                    len(out_script.splitlines()), 1000
                )

            report_cmd = [
                str(perf_path),
                "report",
                "--stdio",
                "-i",
                str(perf_data_path),
                "--sort",
                "symbol",
            ]
            rc_report, out_report, err_report = _run_command(
                report_cmd, cwd=collector_dir
            )
            data["perf_report_exit_code"] = rc_report
            data["perf_report_stderr"] = err_report
            if rc_report == 0:
                hotspots = _parse_perf_report(out_report)
                data["hotspots"] = hotspots
            else:
                warnings.append("perf report failed; hotspots unavailable.")

            if flamegraph_dir:
                stackcollapse = flamegraph_dir / "stackcollapse-perf.pl"
                flamegraph_pl = flamegraph_dir / "flamegraph.pl"
                if stackcollapse.exists() and flamegraph_pl.exists():
                    rc_collapse, out_collapse, err_collapse = _run_command(
                        ["perl", str(stackcollapse)],
                        cwd=collector_dir,
                        timeout=300,
                        input_text=perf_script_path.read_text(),
                    )
                    if rc_collapse == 0:
                        folded_path.write_text(out_collapse)
                        raw_files.append(str(folded_path))
                        rc_fg, out_fg, err_fg = _run_command(
                            [
                                "perl",
                                str(flamegraph_pl),
                                "--color",
                                config.flamegraph_colors,
                            ],
                            cwd=collector_dir,
                            timeout=300,
                            input_text=out_collapse,
                        )
                        if rc_fg == 0:
                            flamegraph_path.write_bytes(out_fg.encode())
                            raw_files.append(str(flamegraph_path))
                        else:
                            warnings.append(
                                f"flamegraph.pl failed with exit code {rc_fg}"
                            )
                            data["flamegraph_stderr"] = err_fg
                    else:
                        warnings.append(
                            "stackcollapse-perf.pl failed; flamegraph not generated."
                        )
                        data["stackcollapse_stderr"] = err_collapse
                else:
                    warnings.append(
                        "FlameGraph scripts not found; skipping SVG generation."
                    )
            else:
                warnings.append(
                    "FlameGraph path not configured; skipping SVG generation."
                )

        duration = time.perf_counter() - start
        success = rc == 0 and not errors

        data.update({
            "perf_data_path": str(perf_data_path),
            "perf_script_path": str(perf_script_path),
            "folded_path": str(folded_path),
            "flamegraph_path": str(flamegraph_path),
        })

        return CollectorResult(
            collector_name=self.name,
            success=success,
            duration_seconds=duration,
            data=data,
            errors=errors,
            warnings=warnings,
            raw_files=raw_files,
        )


def generate_differential_flamegraph(
    baseline_folded: Path,
    target_folded: Path,
    flamegraph_dir: Path,
    output_path: Path,
    color: str = "hot",
) -> Path:
    """Generate a differential flamegraph SVG using difffolded + flamegraph.pl."""
    diff_script = flamegraph_dir / "difffolded.pl"
    flamegraph_pl = flamegraph_dir / "flamegraph.pl"

    if not diff_script.exists():
        raise FileNotFoundError(
            f"difffolded.pl not found under {flamegraph_dir}"
        )
    if not flamegraph_pl.exists():
        raise FileNotFoundError(
            f"flamegraph.pl not found under {flamegraph_dir}"
        )

    rc_diff, out_diff, err_diff = _run_command(
        ["perl", str(diff_script), str(baseline_folded), str(target_folded)],
        cwd=flamegraph_dir,
        timeout=300,
    )
    if rc_diff != 0:
        raise RuntimeError(f"difffolded.pl failed: {err_diff or out_diff}")

    rc_fg, out_fg, err_fg = _run_command(
        ["perl", str(flamegraph_pl), "--color", color, "--diff"],
        cwd=flamegraph_dir,
        timeout=300,
        input_text=out_diff,
    )
    if rc_fg != 0:
        raise RuntimeError(f"flamegraph.pl --diff failed: {err_fg or out_fg}")

    output_path.write_bytes(out_fg.encode())
    return output_path
