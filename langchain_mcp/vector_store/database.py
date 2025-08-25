import sqlite3
from typing import Optional, List, Tuple


class Database:
    def __init__(self, db_path: str = "files.db"):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path)
        self._create_table()

    def _create_table(self):
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    directory TEXT NOT NULL,
                    content TEXT NOT NULL
                )
                """
            )

    def create_file(self, directory: str, content: str) -> int:
        """
        Insert a new file record.

        Returns:
            The ID of the inserted record.
        """
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO files (directory, content) VALUES (?, ?)",
                (directory, content),
            )
            return cursor.lastrowid

    def get_file_by_id(self, file_id: int) -> Optional[Tuple[int, str, str]]:
        """
        Retrieve a single file record by ID.
        """
        cursor = self.connection.execute("SELECT * FROM files WHERE id = ?", (file_id,))
        return cursor.fetchone()

    def get_all_files(self) -> List[Tuple[int, str, str]]:
        """
        Retrieve all file records.
        """
        cursor = self.connection.execute("SELECT * FROM files")
        return cursor.fetchall()

    def update_file(self, file_id: int, directory: str, content: str) -> bool:
        """
        Update a file record by ID.

        Returns:
            True if a row was updated, False otherwise.
        """
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE files SET directory = ?, content = ? WHERE id = ?",
                (directory, content, file_id),
            )
            return cursor.rowcount > 0

    def delete_file(self, file_id: int) -> bool:
        """
        Delete a file record by ID.

        Returns:
            True if a row was deleted, False otherwise.
        """
        with self.connection:
            cursor = self.connection.execute(
                "DELETE FROM files WHERE id = ?", (file_id,)
            )
            return cursor.rowcount > 0

    def close(self):
        """
        Close the database connection.
        """
        self.connection.close()
