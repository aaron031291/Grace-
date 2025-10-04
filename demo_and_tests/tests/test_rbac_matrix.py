"""
Role Matrix RBAC Tests for Grace Backend
"""
import pytest
from backend.auth import verify_token, create_access_token
class User:
    def __init__(self, id, username, role):
        self.id = id
        self.username = username
        self.role = role

# Example role matrix
ROLE_MATRIX = {
    "admin": ["read", "write", "delete", "manage_users"],
    "editor": ["read", "write"],
    "viewer": ["read"],
}

@pytest.mark.parametrize("role,action,expected", [
    ("admin", "manage_users", True),
    ("admin", "delete", True),
    ("editor", "write", True),
    ("editor", "delete", False),
    ("viewer", "read", True),
    ("viewer", "write", False),
])
def test_role_matrix_permissions(role, action, expected):
    allowed = action in ROLE_MATRIX.get(role, [])
    assert allowed == expected

# Example: test JWT role claims
@pytest.mark.parametrize("role", ["admin", "editor", "viewer"])
def test_jwt_role_claims(role):
    user = User(id=1, username="test", role=role)
    token = create_access_token({"sub": user.username, "role": role})
    payload = verify_token(token)
    assert payload["role"] == role
