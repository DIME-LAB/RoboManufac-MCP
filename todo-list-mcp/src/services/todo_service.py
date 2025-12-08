"""
TodoService.py

This service implements the core business logic for managing todos.
It acts as an intermediary between the data model and the database,
handling all CRUD operations and search functionality.

WHY A SERVICE LAYER?
- Separates business logic from database operations
- Provides a clean API for the application to work with
- Makes it easier to change the database implementation later
- Encapsulates complex operations into simple method calls
"""
from typing import Optional
from src.models.todo import (
    Todo, 
    create_todo, 
    CreateTodoSchema, 
    UpdateTodoSchema
)
from src.services.database_service import database_service


class TodoService:
    """
    TodoService Class
    
    This service follows the repository pattern to provide a clean
    interface for working with todos. It encapsulates all database
    operations and business logic in one place.
    """
    
    def create_todo(self, data: CreateTodoSchema) -> Todo:
        """
        Create a new todo
        
        This method:
        1. Uses the factory function to create a new Todo object
        2. Persists it to the database
        3. Returns the created Todo
        
        Args:
            data: Validated input data (title and description)
        
        Returns:
            The newly created Todo
        """
        # Use the factory function to create a Todo with proper defaults
        todo = create_todo(data)
        
        # Get the database instance
        db = database_service.get_db()
        
        # Prepare the SQL statement for inserting a new todo
        db.execute('''
            INSERT INTO todos (id, title, description, completedAt, skippedAt, createdAt, updatedAt)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            todo.id,
            todo.title,
            todo.description,
            todo.completed_at,
            todo.skipped_at,
            todo.created_at,
            todo.updated_at
        ))
        db.commit()
        
        # Return the created todo
        return todo
    
    def create_todos(self, todos_data: list) -> list[Todo]:
        """
        Create multiple todos at once
        
        Args:
            todos_data: List of validated todo data (title and description)
        
        Returns:
            List of newly created Todos
        """
        created_todos = []
        db = database_service.get_db()
        
        for data in todos_data:
            todo = create_todo(data)
            db.execute('''
                INSERT INTO todos (id, title, description, completedAt, skippedAt, createdAt, updatedAt)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                todo.id,
                todo.title,
                todo.description,
                todo.completed_at,
                todo.skipped_at,
                todo.created_at,
                todo.updated_at
            ))
            created_todos.append(todo)
        
        db.commit()
        return created_todos
    
    def get_todo(self, id: str) -> Optional[Todo]:
        """
        Get a todo by ID
        
        This method:
        1. Queries the database for a todo with the given ID
        2. Converts the database row to a Todo object if found
        
        Args:
            id: The UUID of the todo to retrieve
        
        Returns:
            The Todo if found, None otherwise
        """
        db = database_service.get_db()
        
        # Use parameterized query to prevent SQL injection
        cursor = db.execute('SELECT * FROM todos WHERE id = ?', (id,))
        row = cursor.fetchone()
        
        # Return None if no todo was found
        if not row:
            return None
        
        # Convert the database row to a Todo object
        return self._row_to_todo(row)
    
    def get_all_todos(self) -> list[Todo]:
        """
        Get all todos
        
        This method returns all todos in the database without filtering.
        
        Returns:
            List of all Todos
        """
        db = database_service.get_db()
        cursor = db.execute('SELECT * FROM todos')
        rows = cursor.fetchall()
        
        # Convert each database row to a Todo object
        return [self._row_to_todo(row) for row in rows]
    
    def get_active_todos(self) -> list[Todo]:
        """
        Get all active (non-completed) todos
        
        This method returns only todos that haven't been marked as completed.
        A todo is considered active when its completedAt field is NULL.
        
        Returns:
            List of active Todos
        """
        db = database_service.get_db()
        cursor = db.execute('SELECT * FROM todos WHERE completedAt IS NULL')
        rows = cursor.fetchall()
        
        # Convert each database row to a Todo object
        return [self._row_to_todo(row) for row in rows]
    
    def update_todo(self, data: UpdateTodoSchema) -> Optional[Todo]:
        """
        Update a todo
        
        This method:
        1. Checks if the todo exists
        2. Updates the specified fields
        3. Returns the updated todo
        
        Args:
            data: The update data (id required, title/description optional)
        
        Returns:
            The updated Todo if found, None otherwise
        """
        # First check if the todo exists
        todo = self.get_todo(data.id)
        if not todo:
            return None
        
        # Create a timestamp for the update
        from datetime import datetime
        updated_at = datetime.utcnow().isoformat()
        
        db = database_service.get_db()
        
        # Update with new values or keep existing ones if not provided
        db.execute('''
            UPDATE todos
            SET title = ?, description = ?, updatedAt = ?
            WHERE id = ?
        ''', (
            data.title if data.title else todo.title,
            data.description if data.description else todo.description,
            updated_at,
            todo.id
        ))
        db.commit()
        
        # Return the updated todo
        return self.get_todo(todo.id)
    
    def complete_todo(self, id: str) -> Optional[Todo]:
        """
        Mark a todo as completed
        
        This method:
        1. Checks if the todo exists
        2. Sets the completedAt timestamp to the current time
        3. Clears the skippedAt timestamp (completion overwrites skipped status)
        4. Returns the updated todo
        
        Args:
            id: The UUID of the todo to complete
        
        Returns:
            The updated Todo if found, None otherwise
        """
        # First check if the todo exists
        todo = self.get_todo(id)
        if not todo:
            return None
        
        # Create a timestamp for the completion and update
        from datetime import datetime
        now = datetime.utcnow().isoformat()
        
        db = database_service.get_db()
        
        # Set the completedAt timestamp and clear skippedAt (completion overwrites skipped)
        db.execute('''
            UPDATE todos
            SET completedAt = ?, skippedAt = NULL, updatedAt = ?
            WHERE id = ?
        ''', (now, now, id))
        db.commit()
        
        # Return the updated todo
        return self.get_todo(id)
    
    def skip_todos(self, ids: list[str]) -> list[Todo]:
        """
        Mark one or more todos as skipped (only non-completed ones)
        
        This method:
        1. Checks which todos exist and are not completed
        2. Sets the skippedAt timestamp for those todos
        3. Returns the list of skipped todos
        
        Args:
            ids: List of UUIDs of todos to skip
        
        Returns:
            List of skipped Todos (only non-completed ones)
        """
        from datetime import datetime
        now = datetime.utcnow().isoformat()
        
        skipped_todos = []
        db = database_service.get_db()
        
        for todo_id in ids:
            todo = self.get_todo(todo_id)
            # Only skip if todo exists and is not completed
            if todo and not todo.completed:
                db.execute('''
                    UPDATE todos
                    SET skippedAt = ?, updatedAt = ?
                    WHERE id = ? AND completedAt IS NULL
                ''', (now, now, todo_id))
                skipped_todos.append(self.get_todo(todo_id))
        
        db.commit()
        return skipped_todos
    
    def delete_todo(self, id: str) -> bool:
        """
        Delete a todo
        
        This method removes a todo from the database permanently.
        
        Args:
            id: The UUID of the todo to delete
        
        Returns:
            True if deleted, False if not found or not deleted
        """
        db = database_service.get_db()
        cursor = db.execute('DELETE FROM todos WHERE id = ?', (id,))
        db.commit()
        
        # Check if any rows were affected
        return cursor.rowcount > 0
    
    def clear_all_todos(self) -> int:
        """
        Clear all todos from the database
        
        This method removes all todos from the database permanently.
        
        Returns:
            Number of todos that were deleted
        """
        db = database_service.get_db()
        cursor = db.execute('DELETE FROM todos')
        db.commit()
        
        # Return the number of rows deleted
        return cursor.rowcount
    
    def search_by_title(self, title: str) -> list[Todo]:
        """
        Search todos by title
        
        This method performs a case-insensitive partial match search
        on todo titles.
        
        Args:
            title: The search term to look for in titles
        
        Returns:
            List of matching Todos
        """
        # Add wildcards to the search term for partial matching
        search_term = f'%{title}%'
        
        db = database_service.get_db()
        
        # COLLATE NOCASE makes the search case-insensitive
        cursor = db.execute('SELECT * FROM todos WHERE title LIKE ? COLLATE NOCASE', (search_term,))
        rows = cursor.fetchall()
        
        return [self._row_to_todo(row) for row in rows]
    
    def search_by_date(self, date_str: str) -> list[Todo]:
        """
        Search todos by date
        
        This method finds todos created on a specific date.
        It matches the start of the ISO string with the given date.
        
        Args:
            date_str: The date to search for in YYYY-MM-DD format
        
        Returns:
            List of matching Todos
        """
        # Add wildcard to match the time portion of ISO string
        date_pattern = f'{date_str}%'
        
        db = database_service.get_db()
        cursor = db.execute('SELECT * FROM todos WHERE createdAt LIKE ?', (date_pattern,))
        rows = cursor.fetchall()
        
        return [self._row_to_todo(row) for row in rows]
    
    def summarize_active_todos(self) -> str:
        """
        Generate a summary of active todos
        
        This method creates a markdown-formatted summary of all active todos.
        
        WHY RETURN FORMATTED STRING?
        - Provides ready-to-display content for the MCP client
        - Encapsulates formatting logic in the service
        - Makes it easy for LLMs to present a readable summary
        
        Returns:
            Markdown-formatted summary string
        """
        active_todos = self.get_active_todos()
        
        # Handle the case when there are no active todos
        if len(active_todos) == 0:
            return "No active todos found."
        
        # Create a bulleted list of todo titles
        summary = '\n'.join(f'- {todo.title}' for todo in active_todos)
        return f"# Active Todos Summary\n\nThere are {len(active_todos)} active todos:\n\n{summary}"
    
    def _row_to_todo(self, row: tuple) -> Todo:
        """
        Helper to convert a database row to a Todo object
        
        This private method handles the conversion between the database
        representation and the application model.
        
        WHY SEPARATE THIS LOGIC?
        - Avoids repeating the conversion code in multiple methods
        - Creates a single place to update if the model changes
        - Isolates database-specific knowledge from the rest of the code
        
        Args:
            row: The database row data (tuple from sqlite3)
        
        Returns:
            A properly formatted Todo object
        """
        # SQLite returns rows as tuples: (id, title, description, completedAt, skippedAt, createdAt, updatedAt)
        # Handle both old schema (6 columns) and new schema (7 columns)
        if len(row) == 6:
            # Old schema without skippedAt
            return Todo(
                id=row[0],
                title=row[1],
                description=row[2],
                completed_at=row[3],
                skipped_at=None,
                created_at=row[4],
                updated_at=row[5],
                completed=row[3] is not None,
                skipped=False
            )
        else:
            # New schema with skippedAt
            return Todo(
                id=row[0],
                title=row[1],
                description=row[2],
                completed_at=row[3],
                skipped_at=row[4],
                created_at=row[5],
                updated_at=row[6],
                completed=row[3] is not None,  # Computed from completedAt
                skipped=row[4] is not None  # Computed from skippedAt
            )


# Create a singleton instance for use throughout the application
todo_service = TodoService()

