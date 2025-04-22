import os
from typing import List, Dict, Optional
import pathspec  # You'll need to install this: pip install pathspec


class DirectoryFileReader:
    """
    A class that traverses directories to read content from specified file types,
    respecting .gitignore rules.
    """

    def __init__(
        self,
        file_extensions: List[str] = [
            "py",
            "dart",
            "ts",
            "js",
            "jsx",
            "tsx",
            "php",
            "yaml",
            "md",
        ],
        directory: Optional[str] = None,
        max_depth: int = -1,
        respect_gitignore: bool = True,
    ):
        """
        Initialize the DirectoryFileReader.

        Args:
            file_extensions: List of file extensions to include (without the dot)
            directory: Root directory to start from (defaults to current working directory)
            max_depth: Maximum directory depth to traverse (-1 for unlimited)
            respect_gitignore: Whether to respect .gitignore rules
        """
        self.file_extensions = [ext.lower().lstrip(".") for ext in file_extensions]
        self.directory = directory or os.getcwd()
        self.max_depth = max_depth
        self.respect_gitignore = respect_gitignore
        self.gitignore_spec = self._load_gitignore() if respect_gitignore else None

    def _load_gitignore(self) -> Optional[pathspec.PathSpec]:
        """
        Load gitignore patterns from .gitignore files.

        Returns:
            PathSpec object with gitignore patterns, or None if no .gitignore found
        """
        gitignore_path = os.path.join(self.directory, ".gitignore")
        print(f"Loaded gitignore from {gitignore_path}")

        if not os.path.exists(gitignore_path):
            return None

        try:
            with open(gitignore_path, "r") as gitignore_file:
                gitignore_content = gitignore_file.read()

            # Parse the gitignore content and create a PathSpec object
            return pathspec.PathSpec.from_lines(
                pathspec.patterns.GitWildMatchPattern, gitignore_content.splitlines()
            )
        except Exception as e:
            print(f"Error loading .gitignore: {str(e)}")
            return None

    def collect_files(self) -> Dict[str, str]:
        """
        Traverse directories and collect content from files with specified extensions.

        Returns:
            Dictionary mapping file paths to their text content
        """
        result = {}
        self._traverse_directory(self.directory, result, current_depth=0)
        return result

    def get_combined_text(self, separator: str = "\n\n") -> str:
        """
        Get all text content from files combined into a single string.

        Args:
            separator: String to use between content from different files
                      (defaults to double newline)

        Returns:
            Combined string of all text content
        """
        files_content = self.collect_files()

        # Prepare content chunks with file path headers
        content_chunks = []
        for file_path, content in files_content.items():
            # Add file path as a header
            relative_path = os.path.relpath(file_path, self.directory)
            content_chunks.append(f"File: {relative_path}\n{content}")

        # Join all content with the separator
        return separator.join(content_chunks)

    def get_raw_combined_text(self, separator: str = "\n") -> str:
        """
        Get all text content from files combined into a single string without file headers.

        Args:
            separator: String to use between content from different files
                      (defaults to newline)

        Returns:
            Combined string of all text content without file headers
        """
        files_content = self.collect_files()
        return separator.join(files_content.values())

    def _is_ignored(self, path: str) -> bool:
        """
        Check if a file or directory should be ignored according to .gitignore rules.

        Args:
            path: Path to check

        Returns:
            True if the path should be ignored, False otherwise
        """
        if not self.respect_gitignore or self.gitignore_spec is None:
            return False

        # Get the relative path from the base directory
        rel_path = os.path.relpath(path, self.directory)
        # Normalize for pathspec matching
        rel_path = rel_path.replace(os.sep, "/")

        # Check if the path matches any gitignore pattern
        return self.gitignore_spec.match_file(rel_path)

    def _traverse_directory(
        self, current_dir: str, result: Dict[str, str], current_depth: int
    ) -> None:
        """
        Helper method to recursively traverse directories.

        Args:
            current_dir: Current directory being traversed
            result: Dictionary to store results
            current_depth: Current depth of traversal
        """
        # Check if we've exceeded the maximum depth
        if self.max_depth >= 0 and current_depth > self.max_depth:
            return

        # Check if this directory should be ignored
        if self._is_ignored(current_dir):
            return

        try:
            # List all items in the current directory
            items = os.listdir(current_dir)

            for item in items:
                item_path = os.path.join(current_dir, item)

                # Skip if the item is ignored by gitignore rules
                if self._is_ignored(item_path):
                    continue

                # If it's a directory, traverse it recursively
                if os.path.isdir(item_path):
                    self._traverse_directory(item_path, result, current_depth + 1)

                # If it's a file with a matching extension, read its content
                elif os.path.isfile(item_path):
                    file_ext = os.path.splitext(item)[1].lower().lstrip(".")
                    if file_ext in self.file_extensions:
                        try:
                            with open(item_path, "r", encoding="utf-8") as file:
                                result[item_path] = file.read()
                        except Exception as e:
                            # Handle file reading errors
                            result[item_path] = f"Error reading file: {str(e)}"

        except PermissionError:
            # Handle permission errors
            pass
        except Exception as e:
            # Handle other errors during directory traversal
            print(f"Error traversing directory {current_dir}: {str(e)}")
