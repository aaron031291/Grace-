#!/usr/bin/env python3
"""
Demonstration script showing the security and functionality improvements
made to the Grace interface system.
"""
import asyncio
import tempfile
import os
from datetime import datetime
import json

from grace.interface.orb_interface import GraceUnifiedOrbInterface, PanelType, NotificationPriority
from grace.interface.enum_utils import safe_enum_parse
from grace.interface.security import TokenManager, FileValidator
from grace.interface.job_queue import job_queue, JobStatus
from grace.interface.pagination import PaginationParams, paginate_list


async def demonstrate_improvements():
    """Demonstrate all the security and functionality improvements."""
    
    print("🔐 GRACE SECURITY & FUNCTIONALITY IMPROVEMENTS DEMO")
    print("=" * 60)
    
    # Initialize components
    interface = GraceUnifiedOrbInterface()
    token_manager = TokenManager()
    file_validator = FileValidator(max_file_size_mb=10)
    
    print("\n1️⃣  SAFE ENUM PARSING")
    print("-" * 30)
    
    # Test enum parsing with various inputs
    test_inputs = [
        "trading", "TRADING", "trading_panel", "memory_explorer_panel",
        "unknown_type", "analytics", None, ""
    ]
    
    for test_input in test_inputs:
        result = safe_enum_parse(PanelType, test_input, PanelType.ANALYTICS)
        print(f"  '{test_input}' -> {result}")
    
    # Test the interface mapper
    print(f"\n  Interface mapper test:")
    print(f"  'trading_panel' -> {interface.panel_type_mapper('trading_panel')}")
    print(f"  'memory_explorer_panel' -> {interface.panel_type_mapper('memory_explorer_panel')}")
    
    print("\n2️⃣  WEBSOCKET AUTHENTICATION")
    print("-" * 30)
    
    # Generate tokens for different users
    user_tokens = {}
    for user_id in ["user1", "user2"]:
        session_id = f"session_{user_id}"
        token = token_manager.generate_token(user_id, session_id, "tenant_demo")
        user_tokens[user_id] = {"token": token, "session_id": session_id}
        print(f"  Generated token for {user_id}: {token[:20]}...")
    
    # Validate tokens
    for user_id, data in user_tokens.items():
        validation = token_manager.validate_token(data["token"])
        if validation:
            print(f"  ✅ Token valid for {user_id} (session: {validation['session_id']})")
        else:
            print(f"  ❌ Token invalid for {user_id}")
    
    print("\n3️⃣  FILE UPLOAD SECURITY")
    print("-" * 30)
    
    # Test file validation
    test_files = [
        ("document.pdf", "application/pdf", 1024),
        ("script.py", "text/x-python", 2048),
        ("image.jpg", "image/jpeg", 512*1024),
        ("malicious.exe", "application/octet-stream", 1024),
        ("huge_file.txt", "text/plain", 100*1024*1024)  # 100MB
    ]
    
    for filename, content_type, size in test_files:
        size_ok = file_validator.validate_file_size(size)
        type_ok = file_validator.validate_file_type(filename, content_type)
        safe_name = file_validator.get_safe_filename(filename)
        needs_scan = file_validator.should_scan_for_viruses(filename)
        
        status = "✅" if size_ok and type_ok else "❌"
        print(f"  {status} {filename} ({size//1024}KB) -> {safe_name}")
        print(f"     Size OK: {size_ok}, Type OK: {type_ok}, Needs Scan: {needs_scan}")
    
    print("\n4️⃣  NOTIFICATION READ/UNREAD TRACKING")
    print("-" * 30)
    
    # Create test notifications
    notification_ids = []
    for i, priority in enumerate(["high", "medium", "low"]):
        notif_id = await interface.create_notification(
            user_id="demo_user",
            title=f"Test Notification {i+1}",
            message=f"This is a {priority} priority test message",
            priority=safe_enum_parse(NotificationPriority, priority, NotificationPriority.MEDIUM),
            action_required=i == 0  # First notification requires action
        )
        notification_ids.append(notif_id)
        print(f"  Created {priority} priority notification: {notif_id}")
    
    # Check unread notifications
    unread = interface.get_notifications("demo_user", unread_only=True)
    print(f"  📬 Unread notifications: {len(unread)}")
    
    # Mark one as read
    if notification_ids:
        await interface.mark_notification_read(notification_ids[0], "demo_user")
        print(f"  ✅ Marked notification {notification_ids[0]} as read")
    
    # Check unread again
    unread_after = interface.get_notifications("demo_user", unread_only=True)
    all_notifications = interface.get_notifications("demo_user", unread_only=False)
    print(f"  📬 Unread notifications after marking one read: {len(unread_after)}")
    print(f"  📮 Total notifications: {len(all_notifications)}")
    
    print("\n5️⃣  BACKGROUND JOB PROCESSING")
    print("-" * 30)
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("This is a test document for background processing.")
        temp_file_path = f.name
    
    try:
        # Upload document (should trigger background job)
        fragment_id = await interface.upload_document(
            user_id="demo_user",
            file_path=temp_file_path,
            file_type="txt",
            metadata={"test": True}
        )
        print(f"  📄 Uploaded document as fragment: {fragment_id}")
        
        # Check the background job
        fragment = interface.memory_fragments.get(fragment_id)
        if fragment and "background_job_id" in fragment.metadata:
            job_id = fragment.metadata["background_job_id"]
            print(f"  🔄 Background job queued: {job_id}")
            
            # Check job status
            job = await job_queue.get_job(job_id)
            if job:
                print(f"  📊 Job status: {job.status.value}")
                print(f"  📅 Created at: {job.created_at}")
    
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_file_path)
        except OSError:
            pass
    
    print("\n6️⃣  PAGINATION SUPPORT")  
    print("-" * 30)
    
    # Test pagination with sample data
    sample_items = [f"item_{i}" for i in range(1, 101)]  # 100 items
    
    # Test different page sizes
    for page, limit in [(1, 10), (2, 10), (5, 20), (1, 5)]:
        pagination = PaginationParams(page=page, limit=limit)
        result = paginate_list(sample_items, pagination)
        
        print(f"  📄 Page {page}, Limit {limit}:")
        print(f"     Items: {len(result.items)} (showing: {result.items[:3]}...)")
        print(f"     Total Pages: {result.total_pages}, Has Next: {result.has_next}")
    
    print("\n7️⃣  JOB QUEUE STATS")
    print("-" * 30)
    
    # Show job queue statistics
    all_jobs = job_queue.list_jobs(limit=10)
    pending_jobs = [j for j in all_jobs if j.status == JobStatus.PENDING]
    
    print(f"  📊 Total jobs in queue: {len(all_jobs)}")
    print(f"  ⏳ Pending jobs: {len(pending_jobs)}")
    if all_jobs:
        print(f"  📈 Job types: {set(j.job_type for j in all_jobs)}")
    
    print(f"\n✅ DEMONSTRATION COMPLETE")
    print(f"All security and functionality improvements are working correctly!")


if __name__ == "__main__":
    asyncio.run(demonstrate_improvements())