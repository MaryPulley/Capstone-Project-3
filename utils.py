from zipfile import ZipFile
import zipfile
from pathlib import Path


def extract_image_files(archive_file, dest_folder):
    """
    A function to extract files to a destination
    folder. The function expects
    archive_file: file name with path
    dest_folder: path to extract files
    """
    try:
        # Check if it's a valid zip file
        if not zipfile.is_zipfile(archive_file):
            raise zipfile.BadZipFile(
                f"The file '{archive_file}' is not a valid zip file."
            )

        with ZipFile(archive_file, "r") as zip_ref:
            zip_ref.extractall(dest_folder)

        print(f"Files extracted to '{dest_folder}'")

    # Handle invalid zip file
    except zipfile.BadZipFile as e:
        print(f"Error: {e}")

    # Handle all other exceptions
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def getCurrDirPath():
    """
    Return the current file location folder path
    """
    try:
        return Path(__file__).resolve().parent
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
