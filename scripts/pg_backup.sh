#!/bin/bash
# Postgres PITR & Nightly Snapshot Script
# Usage: ./scripts/pg_backup.sh

set -e

PG_CONTAINER=grace_postgres
BACKUP_DIR=/backups/postgres
WAL_DIR=/backups/postgres/wal
DATE=$(date +"%Y-%m-%d_%H-%M-%S")

# Create backup directories if not exist
mkdir -p $BACKUP_DIR $WAL_DIR

# Base backup (nightly)
docker exec $PG_CONTAINER pg_basebackup -D $BACKUP_DIR/$DATE -F tar -z -P -X fetch --wal-method=stream

# Archive WAL files
docker exec $PG_CONTAINER bash -c 'cp /var/lib/postgresql/data/pg_wal/* $WAL_DIR/'

# Clean up old backups (keep 7 days)
find $BACKUP_DIR -type d -mtime +7 -exec rm -rf {} +
find $WAL_DIR -type f -mtime +7 -delete

echo "Postgres PITR base backup and WAL archiving complete."
