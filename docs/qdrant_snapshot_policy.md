# Qdrant Snapshot & Restore Policy

## Snapshot (Manual or Scheduled)

# Create a snapshot of all collections
curl -X POST "http://localhost:6333/collections/snapshots" -H "Content-Type: application/json"

# List all snapshots
curl -X GET "http://localhost:6333/collections/snapshots" -H "Content-Type: application/json"

# Download a snapshot (replace <collection> and <snapshot_name>)
curl -O "http://localhost:6333/collections/<collection>/snapshots/<snapshot_name>"

## Qdrant Snapshot/Export Policy

Qdrant is used for vector storage and similarity search. To ensure reliability and disaster recovery, follow this snapshot and export policy:

## Automated Snapshots
- Qdrant stores data in `/qdrant/storage` (see `docker-compose.yml`).
- Schedule nightly snapshots of the storage directory:

```bash
# Example cron job (host)
0 2 * * * tar czf /backups/qdrant_$(date +\%F).tar.gz -C /path/to/qdrant/storage .
```

## Manual Export
- To export collections manually:

```bash
curl -X POST "http://localhost:6333/collections/<collection_name>/snapshots"
```
- Download snapshot from `/qdrant/storage/collections/<collection_name>/snapshots/`.

## Restore Procedure
- To restore, stop Qdrant, replace `/qdrant/storage` with snapshot contents, then restart Qdrant.

## Retention Policy
- Keep daily snapshots for 7 days, weekly for 4 weeks, monthly for 6 months.
- Store offsite for DR compliance.

## References
- [Qdrant Backup & Restore Docs](https://qdrant.tech/documentation/backup/)

---

**See also:** `docker-compose.yml` for Qdrant volume config.
## Restore
# Upload and restore a snapshot to a collection
curl -X POST "http://localhost:6333/collections/<collection>/snapshots/upload" \
     -F "snapshot=@/path/to/snapshot_file"

## Policy
- Take daily snapshots of all collections.
## Example Cron (host)
0 2 * * * curl -X POST "http://localhost:6333/collections/snapshots"

## Restore Documentation
1. Download or upload the desired snapshot file.
2. Use the restore endpoint above to restore to the target collection.
3. Validate collection health after restore.
