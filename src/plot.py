import pandas as pd
import matplotlib.pyplot as plt
import string

def load_and_prepare_data(file_path):
    """
    Load the Excel file and prepare a single DataFrame with an added column for model and version.
    """
    excel_data = pd.ExcelFile(file_path)
    all_data = []

    for sheet_name in excel_data.sheet_names:
        # Load each sheet
        data = excel_data.parse(sheet_name)
        # Add columns to identify model and version
        if "mini" in sheet_name:
            model = "GPT-4o-mini"
        else:
            model = "GPT-4o"

        if "baseline" in sheet_name:
            version = "Baseline"
        elif "ver_a" in sheet_name:
            version = "Version A"
        elif "ver_b" in sheet_name:
            version = "Version B"
        elif "ver_c" in sheet_name:
            version = "Version C"
        else:
            version = "Unknown"

        data["Model"] = model
        data["Version"] = version
        all_data.append(data)

    # Concatenate all data into a single DataFrame
    return pd.concat(all_data, ignore_index=True)

def validate_file_name(file_name):
    """
    Validate the file name for saving plots to ensure it has a valid extension.
    """
    valid_extensions = {"png", "jpg", "jpeg", "pdf", "svg", "eps", "tif", "tiff", "webp"}
    if "." not in file_name or file_name.split(".")[-1] not in valid_extensions:
        raise ValueError(
            f"Invalid file extension. Supported formats: {', '.join(valid_extensions)}"
        )

def plot_metric(data, metric, title, save_path=None, file_name=None):
    """
    Plot a specific metric (e.g., precision, recall, f1_score, accuracy) across models and versions per file.
    Optionally save the plot to the specified path with the given file name.
    """
    # Filter data to include only GPT-4o-mini and Baseline versions
    data = data[data["Model"] == "GPT-4o-mini"]

    files = data["index"].unique()
    file_labels = list(string.ascii_uppercase[:len(files)])
    file_mapping = dict(zip(files, file_labels))

    plt.figure(figsize=(12, 6))

    for (model, version), group_data in data.groupby(["Model", "Version"]):
        group_data = group_data.copy()
        group_data["File_Label"] = group_data["index"].map(file_mapping)
        group_data = group_data.sort_values("File_Label")

        plt.plot(
            group_data["File_Label"],
            group_data[metric],
            marker="o",
            label=f"{model} - {version}"
        )

    plt.title(title, fontsize=14)
    plt.xlabel("Files", fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path and file_name:
        validate_file_name(file_name)
        plt.savefig(f"{save_path}/{file_name}", dpi=300, bbox_inches='tight')

    plt.show()

def plot_total_tokens(data, version_a_additional_token_count, version_b_additional_token_count, version_c_additional_token_count, save_path=None, save_file_name=None):
    """
    Plot total token counts for each version against each other.
    Add additional token counts to the respective versions before plotting.
    Optionally save the plot to the specified path with the given file name.
    """
    data = data[data["Model"] == "GPT-4o-mini"].copy()

    for file_name, additional_tokens in version_a_additional_token_count.items():
        data.loc[(data["Version"] == "Version A") & (data["index"] == file_name), "total_tokens"] += additional_tokens

    for file_name, additional_tokens in version_b_additional_token_count.items():
        data.loc[(data["Version"] == "Version B") & (data["index"] == file_name), "total_tokens"] += additional_tokens

    for file_name, additional_tokens in version_c_additional_token_count.items():
        data.loc[(data["Version"] == "Version C") & (data["index"] == file_name), "total_tokens"] += additional_tokens

    plot_metric(data, "total_tokens", "Total Token Count per File", save_path, save_file_name)

def plot_processing_time(data, version_a_additional_processing_time, version_b_additional_processing_time, version_c_additional_processing_time, save_path=None, save_file_name=None):
    """
    Plot processing time for each version against each other.
    Add additional processing time to the respective versions before plotting.
    Optionally save the plot to the specified path with the given file name.
    """
    data = data[data["Model"] == "GPT-4o-mini"].copy()

    for file_name, additional_time in version_a_additional_processing_time.items():
        data.loc[(data["Version"] == "Version A") & (data["index"] == file_name), "processing_time"] += additional_time

    for file_name, additional_time in version_b_additional_processing_time.items():
        data.loc[(data["Version"] == "Version B") & (data["index"] == file_name), "processing_time"] += additional_time

    for file_name, additional_time in version_c_additional_processing_time.items():
        data.loc[(data["Version"] == "Version C") & (data["index"] == file_name), "processing_time"] += additional_time

    plot_metric(data, "processing_time", "Processing Time per File", save_path, save_file_name)