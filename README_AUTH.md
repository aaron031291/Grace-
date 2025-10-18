# Grace Authentication System

## Overview

Production-ready authentication system with:
- ✅ **Real User Model** with SQLAlchemy
- ✅ **Password Hashing** with passlib/bcrypt
- ✅ **JWT Tokens** with access + refresh tokens
- ✅ **Role-Based Access Control** (RBAC)
- ✅ **Account Security** (lockout, failed login attempts)
- ✅ **Token Revocation** and refresh

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
chmod +x run_server.sh
./run_server.sh
```

The server will start on http://localhost:8000

### 3. Test Authentication

```bash
python test_auth.py
```

## Default Credentials

**Username:** `admin`  
**Password:** `Admin123!`

## API Endpoints

### Authentication

- **POST** `/api/v1/auth/token` - Login (get access + refresh tokens)
- **POST** `/api/v1/auth/login` - Login with JSON body
- **POST** `/api/v1/auth/refresh` - Refresh access token
- **POST** `/api/v1/auth/register` - Register new user
- **GET** `/api/v1/auth/me` - Get current user info
- **POST** `/api/v1/auth/change-password` - Change password
- **POST** `/api/v1/auth/logout` - Logout (revoke refresh token)

### Admin Endpoints (requires admin role)

- **GET** `/api/v1/auth/users` - List all users
- **POST** `/api/v1/auth/roles` - Create new role (superuser only)
- **POST** `/api/v1/auth/users/{user_id}/roles/{role_name}` - Assign role

## Usage Examples

### Login

```bash
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=Admin123!"
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Get Current User

```bash
curl -X GET http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Register New User

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "newuser",
    "email": "user@example.com",
    "password": "SecurePass123!",
    "full_name": "New User"
  }'
```

### Refresh Token

```bash
curl -X POST http://localhost:8000/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "YOUR_REFRESH_TOKEN"
  }'
```

## Role-Based Access Control

### Using Roles in Routes

```python
from fastapi import APIRouter, Depends
from grace.auth.dependencies import require_role, get_current_user
from grace.auth.models import User

router = APIRouter()

# Require admin role
@router.get("/admin-only")
async def admin_endpoint(
    current_user: User = Depends(require_role(["admin"]))
):
    return {"message": "Admin access granted"}

# Require any of multiple roles
@router.get("/moderator-or-admin")
async def moderator_endpoint(
    current_user: User = Depends(require_role(["admin", "moderator"]))
):
    return {"message": "Access granted"}

# Just require authentication
@router.get("/authenticated")
async def authenticated_endpoint(
    current_user: User = Depends(get_current_user)
):
    return {"message": f"Hello {current_user.username}"}
```

## Database Models

### User Model

- `id` - Unique identifier
- `username` - Unique username
- `email` - Unique email
- `hashed_password` - Bcrypt hashed password
- `full_name` - Full name (optional)
- `is_active` - Account status
- `is_verified` - Email verification status
- `is_superuser` - Superuser flag
- `roles` - Many-to-many relationship with roles
- `failed_login_attempts` - Failed login counter
- `locked_until` - Account lock timestamp
- `last_login` - Last successful login

### Role Model

- `id` - Unique identifier
- `name` - Role name (unique)
- `description` - Role description
- `users` - Many-to-many relationship with users

### RefreshToken Model

- `id` - Unique identifier
- `token` - JWT refresh token
- `user_id` - Foreign key to user
- `expires_at` - Expiration timestamp
- `revoked` - Revocation flag
- `device_info` - Device information
- `ip_address` - IP address

## Security Features

### Password Requirements

- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one digit
- Hashed with bcrypt

### Account Lockout

- After 5 failed login attempts
- Locked for 30 minutes
- Automatically unlocked after timeout

### Token Security

- Access tokens expire in 30 minutes
- Refresh tokens expire in 7 days
- Tokens include JWT ID (jti) for tracking
- Refresh tokens can be revoked
- All tokens revoked on password change

## Configuration

### Environment Variables

```bash
# Database URL
DATABASE_URL="postgresql://user:password@localhost/grace_db"
# or
DATABASE_URL="sqlite:///./grace_data.db"

# JWT Secret (generate with: python -c "import secrets; print(secrets.token_urlsafe(32))")
JWT_SECRET_KEY="your-secret-key-here"

# Token expiration
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
```

## API Documentation

Interactive API documentation available at:
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

## Testing

Run the test suite:

```bash
# Make sure server is running first
./run_server.sh

# In another terminal
python test_auth.py
```

## Production Deployment

### Security Checklist

- [ ] Change default admin password
- [ ] Set strong JWT_SECRET_KEY from environment
- [ ] Use PostgreSQL instead of SQLite
- [ ] Enable HTTPS only
- [ ] Configure CORS appropriately
- [ ] Set up rate limiting
- [ ] Enable database backups
- [ ] Monitor failed login attempts
- [ ] Implement email verification
- [ ] Add password reset functionality

### Database Migration

For PostgreSQL:

```bash
# Set database URL
export DATABASE_URL="postgresql://user:password@localhost/grace_db"

# Run server (will auto-create tables)
python -m grace.main
```

## Troubleshooting

### Database locked error (SQLite)

Use PostgreSQL in production or ensure only one process accesses SQLite.

### Token verification fails

Check that JWT_SECRET_KEY is consistent across restarts.

### User not found after creation

Ensure database session is committed properly.

## Support

For issues or questions:
- Check logs in console output
- Review API docs at `/api/docs`
- Ensure all dependencies are installed

---

**Grace AI System** - Production-Ready Authentication 🔐
