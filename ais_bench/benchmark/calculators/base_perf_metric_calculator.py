import re
import csv
import math
import numpy as np
from abc import abstractmethod, ABC

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.global_consts import LOG_LEVEL
from ais_bench.benchmark.utils.logging.exceptions import AISBenchMetricError, AISBenchDumpError
from ais_bench.benchmark.utils.logging.error_codes import CALC_CODES

DEFAULT_STATS = [
    "Average", "Min", "Max", "Median", "P75", "P90", "P99",
]
MAX_STATS_LEN = 8
PERCENTAGE_PATTERN = r"^P(0*[1-9]\d{0,1})$"  # P1 ~ P99
SECOND_TO_MILLISECOND = 1000

def is_legal_percentage_str(stat: str) -> bool:
    """
    Check if a string represents a legal percentage statistic.

    Args:
        stat (str): The statistic string to validate (e.g., "P50", "P99")

    Returns:
        bool: True if the string matches the percentage pattern, False otherwise
    """
    return re.match(PERCENTAGE_PATTERN, stat)


class BasePerfMetricCalculator(ABC):
    """
    Base class for performance metric calculators.

    This abstract base class provides the foundation for calculating various
    performance metrics from benchmark results. It handles statistical analysis,
    unit conversion, and data export functionality.

    Args:
        perf_details (dict, optional): Performance details dictionary containing benchmark results
        stats_list (list, optional): List of statistics to calculate (defaults to DEFAULT_STATS)
    """

    def __init__(self, stats_list: list = DEFAULT_STATS):
        """
        Initialize the performance metric calculator.

        Args:
            perf_details (dict, optional): Performance details dictionary
            stats_list (list, optional): List of statistics to calculate
        """
        self.logger = AISLogger()
        self.logger.debug(f"Initializing BasePerfMetricCalculator with stats_list: {stats_list}")
        self.stats_list = self._validate_stats_list(stats_list or DEFAULT_STATS)
        self.logger.debug(f"Validated stats list: {self.stats_list}")
        self.metrics = {}
        self.common_metrics = {}

    @abstractmethod
    def _init_datas(self, perf_details: dict, max_concurrency: int):
        """
        Initialize data structures for performance calculation.

        This method must be implemented by subclasses to set up the necessary
        data structures for performance metric calculation.

        Args:
            perf_details (dict): Performance details dictionary
            max_concurrency (int): Maximum concurrency value
        """
        pass

    def _validate_stats_list(self, stats_list: list) -> list:
        """
        Validate and filter the statistics list.

        Args:
            stats_list (list): List of statistics to validate

        Returns:
            list: Validated and filtered statistics list
        """
        self.logger.debug(f"Validating stats list: {stats_list}")

        if len(stats_list) > MAX_STATS_LEN:
            self.logger.warning(
                f"Statistics list length exceeds {MAX_STATS_LEN}! Only keeping the first {MAX_STATS_LEN} statistic!"
            )
            stats_list = stats_list[:MAX_STATS_LEN]

        valid_stats = []
        for stat in stats_list:
            if stat not in [
                "Average",
                "Min",
                "Max",
                "Median",
            ] and not is_legal_percentage_str(stat):
                self.logger.warning(f"Unknown statistic: {stat}, will be ignored! Legal statistics examples: {DEFAULT_STATS}")
                continue
            valid_stats.append(stat)

        self.logger.debug(f"Validation completed, valid stats: {valid_stats}")

        if len(valid_stats) == 0:
            self.logger.warning(
                "No valid statistics found, using 'Average' as default."
            )
            valid_stats.append("Average")

        return valid_stats

    def _calculate_statistics(self, data: list, stats_list: list = None) -> dict:
        """
        Calculate statistical metrics from the provided data.

        Args:
            data (list): List of numerical data points
            stats_list (list, optional): List of statistics to calculate

        Returns:
            dict: Dictionary containing calculated statistics
        """
        stats_list = stats_list or self.stats_list
        self.logger.debug(f"Calculating statistics for {len(data) if data else 0} data points with stats: {stats_list}")

        stats = {k: 0 for k in stats_list}

        if not data:
            self.logger.warning("Empty data list, returning empty stats")
            return stats

        # Handle numpy arrays
        if isinstance(data[0], np.ndarray):
            self.logger.debug(f"Handling numpy arrays, concatenating {len(data)} arrays")
            arr = np.concatenate(data)
        else:
            self.logger.debug(f"Converting data to numpy array with length {len(data)}")
            arr = np.array(data)

        for stat in stats_list:
            if stat == "Average":
                stats[stat] = round(float(arr.mean()), 4)
            elif stat == "Min":
                stats[stat] = round(float(arr.min()), 4)
            elif stat == "Max":
                stats[stat] = round(float(arr.max()), 4)
            elif stat == "Median":
                stats[stat] = round(float(np.percentile(arr, 50)), 4)
            elif is_legal_percentage_str(stat):
                stats[stat] = round(float(np.percentile(arr, int(stat[1:]))), 4)

        self.logger.debug(f"Statistics calculation completed: {stats}")

        return stats

    def _process_batch_sizes(self, batch_sizes: list) -> list:
        """
        Process and compress batch sizes to remove duplicates.

        Args:
            batch_sizes (list): List of batch sizes to process

        Returns:
            list: Processed batch sizes with duplicates removed
        """
        if not batch_sizes:
            self.logger.debug("Batch sizes list is empty, skip processing batch size compression.")
            return []

        statistics = []
        count_dict = {}

        for batch_size in batch_sizes:
            if batch_size in count_dict:
                count_dict[batch_size] -= 1
                if count_dict[batch_size] == 0:
                    del count_dict[batch_size]
            else:
                count_dict[batch_size] = batch_size - 1
                statistics.append(batch_size)

        if count_dict:
            self.logger.warning("Batch size compression incomplete: %s", count_dict)

        return statistics

    def _add_units_to_metrics(self, metrics: dict) -> dict:
        """
        Add appropriate units to performance metrics.

        Args:
            metrics (dict): Dictionary containing performance metrics

        Returns:
            dict: Metrics dictionary with units added
        """
        ms = " ms"
        unit_token = " token/s"

        metrics_units_map = {
            "E2EL": ms,
            "TTFT": ms,
            "TPOT": ms,
            "ITL": ms,
            "InputTokens": None,
            "OutputTokens": None,
            "PrefillTokenThroughput": unit_token,
            "OutputTokenThroughput": unit_token,
        }

        for metric, values in metrics.items():
            for stage_name, stage_values in values.items():
                if (
                    metric not in metrics_units_map
                    or metrics_units_map.get(metric) is None
                ):
                    continue
                for key, val in stage_values.items():
                    if key == "N":  # N is the number of requests, no need to add unit
                        continue
                    unit = metrics_units_map.get(metric)
                    if unit == ms:
                        val = round(val * SECOND_TO_MILLISECOND, 4)
                    stage_values[key] = str(val) + unit

        return metrics

    def _add_units_to_common_metrics(self, common_metrics: dict) -> dict:
        """
        Add appropriate units to common metrics.

        Args:
            common_metrics (dict): Dictionary containing common metrics

        Returns:
            dict: Common metrics dictionary with units added
        """
        ms = " ms"
        unit_token = " token/s"

        common_metric_units_map = {
            "Benchmark Duration": ms,
            "Total Requests": None,
            "Failed Requests": None,
            "Success Requests": None,
            "Concurrency": None,
            "Max Concurrency": None,
            "Request Throughput": " req/s",
            "Total Input Tokens": None,
            "Prefill Token Throughput": unit_token,
            "Input Token Throughput": unit_token,
            "Total Output Tokens": None,
            "Output Token Throughput": unit_token,
            "Total Token Throughput": unit_token,
        }

        for metric, value in common_metrics.items():
            for stage_name, stage_value in value.items():
                if (
                    metric not in common_metric_units_map
                    or common_metric_units_map.get(metric) is None
                ):
                    continue
                value[stage_name] = str(stage_value) + common_metric_units_map.get(
                    metric
                )

        return common_metrics

    def _export_to_csv(self, metrics: dict, output_path: str):
        """
        Export performance metrics to CSV file.

        Args:
            metrics (dict): Performance metrics dictionary
            output_path (str): Path to output CSV file

        Raises:
            AISBenchMetricError: If metrics data is empty or invalid
            AISBenchDumpError: If file writing fails
        """
        if not metrics:
            raise AISBenchMetricError(
                CALC_CODES.INVALID_METRIC_DATA,
                "Request level performance metrics data is empty, cannot save to file."
            )

        try:
            first_entry = next(iter(metrics.values()), None)
            if first_entry is None:
                raise AISBenchMetricError(
                    CALC_CODES.INVALID_METRIC_DATA,
                    "Structure of request level performance metrics data is invalid."
                )

            stage_names = list(first_entry.keys())
            headers = list(first_entry[stage_names[0]].keys())

            with open(output_path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["Performance Parameters"] + ["Stage"] + headers)

                for obj_name, values in metrics.items():
                    for stage_name in stage_names:
                        row = (
                            [obj_name]
                            + [stage_name]
                            + [values[stage_name].get(key, "") for key in headers]
                        )
                        writer.writerow(row)

        except (OSError, IOError) as e:
            raise AISBenchDumpError(
                CALC_CODES.DUMPING_RESULT_FAILED,
                f"Failed to write request level performance metrics to csv file '{output_path}': {e}"
            )

    def convert_result(self, result: dict) -> dict:
        """
        Convert raw result data to standardized format.

        Args:
            result (dict): Raw result data dictionary

        Returns:
            dict: Converted result in standardized format
        """
        self.logger.debug(f"Converting result data with keys: {list(result.keys())}")

        remove_keys = [
            "id",
            "start_time",
            "end_time",
            "success",
        ]
        for key in remove_keys:
            result.pop(key, None)
        mapping = {
            "latency": "E2EL",
            "ttft": "TTFT",
            "tpot": "TPOT",
            "itl": "ITL",
            "input_tokens": "InputTokens",
            "output_tokens": "OutputTokens",
            "generate_tokens_speed": "OutputTokenThroughput",
        }

        ans = {mapping_value: [] for mapping_value in mapping.values()}

        # Use dictionary comprehension to populate the values
        for mapping_key, mapping_value in mapping.items():
            if not mapping_key in result:
                self.logger.warning(f"Mapping key {mapping_key} not found in result, skipping")
                continue

            self.logger.debug(f"Processing mapping key: {mapping_key} to {mapping_value}")
            for value in result[mapping_key]:
                if isinstance(value, list):
                    ans[mapping_value].extend(value)
                else:
                    ans[mapping_value].append(value)

        for key in ["ITL"]:
            if not ans[key] or (isinstance(ans[key][0], np.ndarray) and not ans[key][0].any()):
                ans.pop(key)

        for key in ["TTFT", "TPOT"]:
            if math.isclose(sum(ans[key]), 0):
                ans.pop(key)

        return ans

    def _calc_metrics(self):
        """
        Calculate various statistical metrics for all stages.
        """
        self.logger.debug("Starting metrics calculation for all stages")
        for stage_name, _ in self.stage_dict.items():
            self.logger.debug(f"Processing stage: {stage_name}")
            for metric, value in self.result[stage_name].items():
                self.logger.debug(f"Calculating metric: {metric} with data length: {len(value) if value else 0}")
                if value:
                    # Special handling for batch size metrics
                    if metric in {"PrefillBatchsize", "DecoderBatchsize"}:
                        value = self._process_batch_sizes(value)

                    # Calculate statistical metrics
                    stats = self._calculate_statistics(value)
                else:
                    stats = {k: 0 for k in self.stats_list}

                # Store metrics
                if self.metrics.get(metric) is None:
                    self.metrics[metric] = {stage_name: stats}
                else:
                    self.metrics[metric][stage_name] = stats
                    self.logger.debug(f"Stored metrics for {metric} in stage {stage_name}: {stats}")

            # Add count information to all metrics
            for key in self.metrics:
                self.metrics[key][stage_name]["N"] = self.success_count[stage_name]
                self.logger.debug(f"Added count information N={self.success_count[stage_name]} for {key} in stage {stage_name}")
                # TPOT and ITL are average decode latency per request, count requests with decode latency
                if key == "TPOT" or key == "ITL":
                    decode_count = sum(
                        [
                            1
                            for decode_list in self.decode_latencies[stage_name]
                            if np.array(decode_list).any()
                        ]
                    )
                    self.metrics[key][stage_name]["N"] = decode_count
                    self.logger.debug(f"Adjusted count for {key} in stage {stage_name} to {decode_count}")

    def _calc_common_metrics(self):
        """
        Calculate common performance metrics.
        """
        self.logger.debug("Starting common metrics calculation")
        common_metric_names = [
            "Benchmark Duration",
            "Total Requests",
            "Failed Requests",
            "Success Requests",
            "Concurrency",
            "Max Concurrency",
            "Request Throughput",
            "Total Input Tokens",
            "Prefill Token Throughput",
            "Total Generated Tokens",
            "Input Token Throughput",
            "Output Token Throughput",
            "Total Token Throughput",
        ]
        for name in common_metric_names:
            self.common_metrics.setdefault(name, {})

        for stage_name, _ in self.stage_dict.items():
            self.logger.debug(f"Calculating common metrics for stage: {stage_name}")
            self.common_metrics["Benchmark Duration"][stage_name] = round(
                self.infer_time[stage_name] * SECOND_TO_MILLISECOND, 4
            )
            self.logger.debug(f"Stage {stage_name} - Benchmark Duration: {self.common_metrics['Benchmark Duration'][stage_name]} ms")

            self.common_metrics["Total Requests"][stage_name] = self.data_count[stage_name]
            self.common_metrics["Failed Requests"][stage_name] = (
                self.data_count[stage_name] - self.success_count[stage_name]
            )
            self.common_metrics["Success Requests"][stage_name] = self.success_count[stage_name]
            self.logger.debug(f"Stage {stage_name} - Request counts: Total={self.common_metrics['Total Requests'][stage_name]}, Success={self.common_metrics['Success Requests'][stage_name]}, Failed={self.common_metrics['Failed Requests'][stage_name]}")

            if self.common_metrics["Failed Requests"][stage_name] > 0:
                self.logger.warning(
                    "Some requests failed, please check the error logs from responses!"
                )

            # Concurrency calculation (can be overridden by subclasses)
            self.common_metrics["Concurrency"][stage_name] = (
                self._calculate_concurrency(stage_name)
            )
            self.logger.debug(f"Calculated concurrency for {stage_name}: {self.common_metrics['Concurrency'][stage_name]}")
            self.common_metrics["Max Concurrency"][stage_name] = self.max_concurrency
            self.logger.debug(f"Set max concurrency for {stage_name}: {self.max_concurrency}")

            try:
                self.common_metrics["Request Throughput"][stage_name] = round(
                    self.success_count[stage_name] / self.infer_time[stage_name], 4
                )
                self.logger.debug(f"Stage {stage_name} - Request Throughput: {self.common_metrics['Request Throughput'][stage_name]} req/s")
            except ZeroDivisionError:
                self.common_metrics["Request Throughput"][stage_name] = 0
                self.logger.debug(f"Stage {stage_name} - Request Throughput: 0 (infer_time is zero)")

            self.common_metrics["Total Input Tokens"][stage_name] = sum(
                self.result[stage_name]["InputTokens"]
            )
            self.logger.debug(f"Stage {stage_name} - Total Input Tokens: {self.common_metrics['Total Input Tokens'][stage_name]}")

            if (
                self.common_metrics["Total Input Tokens"][stage_name] != 0
                and self.result[stage_name].get("TTFT") is not None
            ):
                self.common_metrics["Prefill Token Throughput"][stage_name] = round(
                    self.common_metrics["Total Input Tokens"][stage_name]
                    / sum(self.result[stage_name]["TTFT"]),
                    4,
                )
                self.logger.debug(f"Stage {stage_name} - Prefill Token Throughput: {self.common_metrics['Prefill Token Throughput'][stage_name]} token/s")
            else:
                self.common_metrics.pop("Prefill Token Throughput", None)
                self.logger.debug(f"Stage {stage_name} - Prefill Token Throughput: Not calculated (insufficient data)")

            self.common_metrics["Total Generated Tokens"][stage_name] = sum(
                self.result[stage_name]["OutputTokens"]
            )
            self.logger.debug(f"Stage {stage_name} - Total Generated Tokens: {self.common_metrics['Total Generated Tokens'][stage_name]}")

            if self.infer_time[stage_name] > 0:
                self.common_metrics["Input Token Throughput"][stage_name] = round(
                    self.common_metrics["Total Input Tokens"][stage_name]
                    / self.infer_time[stage_name],
                    4,
                )
                self.logger.debug(f"Stage {stage_name} - Input Token Throughput: {self.common_metrics['Input Token Throughput'][stage_name]} token/s")

                self.common_metrics["Output Token Throughput"][stage_name] = round(
                    sum(self.result[stage_name]["OutputTokens"])
                    / self.infer_time[stage_name],
                    4,
                )
                self.logger.debug(f"Stage {stage_name} - Output Token Throughput: {self.common_metrics['Output Token Throughput'][stage_name]} token/s")

                self.common_metrics["Total Token Throughput"][stage_name] = round(
                    (
                        self.common_metrics["Total Input Tokens"][stage_name]
                        + sum(self.result[stage_name]["OutputTokens"])
                    )
                    / self.infer_time[stage_name],
                    4,
                )
                self.logger.debug(f"Stage {stage_name} - Total Token Throughput: {self.common_metrics['Total Token Throughput'][stage_name]} token/s")

    def _calculate_concurrency(self, stage_name: str) -> float:
        """
        Calculate concurrency for a given stage.

        Note:
            This method can be overridden by subclasses for custom concurrency calculation
        """
        self.logger.debug(f"Calculating concurrency for stage {stage_name}: sum(E2EL)={sum(self.result[stage_name]['E2EL'])}, infer_time={self.infer_time[stage_name]}")
        return round(
            sum(self.result[stage_name]["E2EL"]) / self.infer_time[stage_name], 4
        )

    def calculate(self):
        """
        Execute the complete calculation process.

        This method orchestrates the calculation of metrics, common metrics,
        and unit conversion.
        """
        self.logger.info("Starting metrics calculation...")
        self.logger.debug(f"Available stages: {list(self.stage_dict.keys())}")
        self._calc_metrics()
        self.logger.info("Starting common metrics calculation...")
        self._calc_common_metrics()
        self.logger.info("Adding units to metrics...")
        self.add_units()
        self.logger.debug(f"Final metrics keys: {list(self.metrics.keys())}")
        self.logger.debug(f"Final common metrics keys: {list(self.common_metrics.keys())}")
        self.logger.info("Performance data calculation completed!")

    def add_units(self):
        """
        Add appropriate units to all metrics.
        """
        self.metrics = self._add_units_to_metrics(self.metrics)
        self.common_metrics = self._add_units_to_common_metrics(self.common_metrics)

    def get_common_res(self) -> dict:
        """
        Get common performance results.

        Returns:
            dict: Dictionary containing common performance metrics
        """
        return {k: v for k, v in self.common_metrics.items() if v is not None}

    def save_performance(self, out_path: str):
        """
        Save performance data to CSV file.

        Args:
            out_path (str): Output file path for the CSV file
        """
        self.logger.debug(f"Saving performance data to CSV file: {out_path}")
        self._export_to_csv(self.metrics, out_path)
        self.logger.debug(f"Performance data successfully saved to {out_path}")

    def generate_curve_html(self, output_path: str):
        """
        Generate density curve distribution visualization HTML for all metrics.

        Creates a comprehensive HTML file with density curve charts (KDE) for each metric
        across all stages, showing the smooth distribution of values.

        Args:
            output_path (str): Path to output HTML file
        """
        if not hasattr(self, 'result') or not self.result:
            self.logger.warning("No result data available for curve generation, skipping.")
            return

        self.logger.debug("Generating density curve HTML visualization...")

        # Collect all metrics and stages
        all_metrics = set()
        all_stages = set()
        for stage_name, stage_data in self.result.items():
            all_stages.add(stage_name)
            all_metrics.update(stage_data.keys())

        if not all_metrics:
            self.logger.warning("No metrics found for curve generation, skipping.")
            return

        # Filter out metrics that shouldn't be visualized as curves
        # (e.g., batch sizes that are already processed)
        metrics_to_visualize = [
            m for m in sorted(all_metrics)
            if m not in {"PrefillBatchsize", "DecoderBatchsize"}
        ]

        if not metrics_to_visualize:
            self.logger.warning("No suitable metrics for curve visualization, skipping.")
            return

        # Calculate number of subplots needed
        num_metrics = len(metrics_to_visualize)
        num_stages = len(all_stages)

        # Create subplots: one row per metric, one column per stage
        fig = make_subplots(
            rows=num_metrics,
            cols=num_stages,
            subplot_titles=[
                f"{metric} - {stage}"
                for metric in metrics_to_visualize
                for stage in sorted(all_stages)
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )

        # Generate density curve for each metric-stage combination
        for metric_idx, metric in enumerate(metrics_to_visualize):
            for stage_idx, stage_name in enumerate(sorted(all_stages)):
                row = metric_idx + 1
                col = stage_idx + 1

                # Get data for this metric and stage
                if stage_name not in self.result or metric not in self.result[stage_name]:
                    self.logger.debug(f"No data for {metric} in stage {stage_name}, skipping.")
                    continue

                data = self.result[stage_name][metric]
                if not data:
                    self.logger.debug(f"Empty data for {metric} in stage {stage_name}, skipping.")
                    continue

                # Flatten data if it contains numpy arrays
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], np.ndarray):
                        flat_data = np.concatenate(data)
                    else:
                        flat_data = np.array(data)
                else:
                    flat_data = np.array(data)

                # Remove invalid values (NaN, inf)
                flat_data = flat_data[np.isfinite(flat_data)]

                if len(flat_data) == 0:
                    self.logger.debug(f"No valid data for {metric} in stage {stage_name}, skipping.")
                    continue

                # Handle single value or all values identical
                unique_values = np.unique(flat_data)
                if len(unique_values) == 1:
                    # Single value case: display as a vertical line with annotation
                    single_value = unique_values[0]
                    # Create a small range around the value for visualization
                    value_range = max(abs(single_value) * 0.1, 1.0) if single_value != 0 else 1.0
                    x_min = single_value - value_range
                    x_max = single_value + value_range

                    # Create a vertical line at the value
                    x_line = [single_value, single_value]
                    y_line = [0, 1]  # Normalized height

                    fig.add_trace(
                        go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode='lines+markers',
                            name=f"{metric} - {stage_name}",
                            showlegend=False,
                            line=dict(
                                color='rgba(55, 128, 191, 0.8)',
                                width=3
                            ),
                            marker=dict(
                                size=10,
                                color='rgba(55, 128, 191, 1.0)',
                                symbol='diamond'
                            ),
                            hovertemplate=f'<b>{metric} - {stage_name}</b><br>' +
                                         f'Value: {single_value:.4f}<br>' +
                                         f'Count: {len(flat_data)}<br>' +
                                         '<i>All values are identical</i><extra></extra>'
                        ),
                        row=row,
                        col=col
                    )

                    # Add annotation showing the value
                    fig.add_annotation(
                        x=single_value,
                        y=0.5,
                        text=f"Value: {single_value:.4f}<br>({len(flat_data)} samples)",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor='rgba(55, 128, 191, 0.8)',
                        bgcolor='rgba(255, 255, 255, 0.9)',
                        bordercolor='rgba(55, 128, 191, 0.8)',
                        borderwidth=1,
                        row=row,
                        col=col
                    )

                    # Update axes
                    fig.update_xaxes(
                        title_text=metric,
                        range=[x_min, x_max],
                        row=row,
                        col=col
                    )

                    if col == 1:
                        fig.update_yaxes(
                            title_text="Normalized",
                            range=[0, 1.1],
                            row=row,
                            col=col
                        )
                    continue

                # Need at least 2 distinct points for KDE
                if len(flat_data) < 2:
                    # Fallback: show as a point
                    single_value = flat_data[0]
                    value_range = max(abs(single_value) * 0.1, 1.0) if single_value != 0 else 1.0
                    x_min = single_value - value_range
                    x_max = single_value + value_range

                    fig.add_trace(
                        go.Scatter(
                            x=[single_value],
                            y=[1],
                            mode='markers',
                            name=f"{metric} - {stage_name}",
                            showlegend=False,
                            marker=dict(
                                size=15,
                                color='rgba(55, 128, 191, 1.0)',
                                symbol='diamond',
                                line=dict(width=2, color='rgba(55, 128, 191, 0.8)')
                            ),
                            hovertemplate=f'<b>{metric} - {stage_name}</b><br>' +
                                         f'Value: {single_value:.4f}<extra></extra>'
                        ),
                        row=row,
                        col=col
                    )

                    fig.update_xaxes(
                        title_text=metric,
                        range=[x_min, x_max],
                        row=row,
                        col=col
                    )

                    if col == 1:
                        fig.update_yaxes(
                            title_text="",
                            range=[0.8, 1.2],
                            row=row,
                            col=col
                        )
                    continue

                try:
                    # Calculate KDE (Kernel Density Estimation)
                    # Handle case where all values are very close (low variance)
                    data_std = flat_data.std()
                    if data_std < 1e-10:
                        # All values are essentially the same, use a small bandwidth
                        kde = stats.gaussian_kde(flat_data)
                        kde.set_bandwidth(kde.factor * 0.1)  # Use smaller bandwidth
                    else:
                        kde = stats.gaussian_kde(flat_data)

                    # Create evaluation points for smooth curve
                    data_min = flat_data.min()
                    data_max = flat_data.max()
                    data_range = data_max - data_min

                    # Extend range slightly for better visualization
                    # Use a minimum range to avoid too narrow plots
                    min_range = max(data_range, abs(data_min) * 0.1, abs(data_max) * 0.1, 1.0)
                    if data_range < min_range * 0.1:
                        # Very small range, extend more
                        x_min = data_min - min_range * 0.2
                        x_max = data_max + min_range * 0.2
                    else:
                        x_min = data_min - 0.1 * data_range
                        x_max = data_max + 0.1 * data_range

                    # Generate smooth curve points
                    num_points = min(500, max(100, len(flat_data) * 2))
                    x_curve = np.linspace(x_min, x_max, num_points)
                    y_curve = kde(x_curve)

                    # Normalize y_curve to make it more visually appealing
                    if y_curve.max() > 0:
                        y_curve = y_curve / y_curve.max()

                    # Create density curve
                    fig.add_trace(
                        go.Scatter(
                            x=x_curve,
                            y=y_curve,
                            mode='lines',
                            name=f"{metric} - {stage_name}",
                            showlegend=False,
                            line=dict(
                                color='rgba(55, 128, 191, 0.8)',
                                width=2
                            ),
                            fill='tozeroy',
                            fillcolor='rgba(55, 128, 191, 0.2)',
                            hovertemplate=f'<b>{metric} - {stage_name}</b><br>' +
                                         'Value: %{x:.4f}<br>' +
                                         'Density: %{y:.4f}<br>' +
                                         f'Samples: {len(flat_data)}<extra></extra>'
                        ),
                        row=row,
                        col=col
                    )

                    # Update x-axis label
                    fig.update_xaxes(
                        title_text=metric,
                        row=row,
                        col=col
                    )

                    # Update y-axis label (only for first column)
                    if col == 1:
                        fig.update_yaxes(
                            title_text="Normalized Density",
                            range=[0, 1.1],
                            row=row,
                            col=col
                        )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to generate KDE curve for {metric} in stage {stage_name}: {e}"
                    )
                    continue

        # Update layout
        fig.update_layout(
            title_text="Performance Metrics Density Distribution Curves (KDE)",
            title_x=0.5,
            height=300 * num_metrics,
            showlegend=False,
            template="plotly_white"
        )

        # Save to HTML
        try:
            fig.write_html(
                output_path,
                include_plotlyjs="cdn",
                config={
                    'scrollZoom': True,
                    'plotGlPixelRatio': 1,
                    'showLink': False,
                    'displaylogo': False,
                    'responsive': True
                },
                auto_open=False,
                full_html=True
            )
            self.logger.info(f"Density curve visualization saved to {output_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save curve HTML to {output_path}: {e}")
