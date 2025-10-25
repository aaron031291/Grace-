#!/usr/bin/env python3
"""
Grace AI - Comprehensive System Verification Script
===================================================
This script initializes the entire Grace system, verifies that all kernels
are correctly wired to their required services, and runs health checks
on all major components.
"""

import sys
import asyncio
import logging

# Add workspace to path
sys.path.insert(0, '/workspaces/Grace-')

from grace.launcher import GraceLauncher

# --- ANSI Color Codes ---
C_RED = '\033[91m'
C_GREEN = '\033[92m'
C_YELLOW = '\033[93m'
C_BLUE = '\033[94m'
C_MAGENTA = '\033[95m'
C_CYAN = '\033[96m'
C_END = '\033[0m'

# Suppress most logging to keep output clean, only show errors
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class SystemVerifier:
    """
    A class to verify the complete wiring and health of the Grace AI system.
    """

    def __init__(self):
        self.launcher = None
        self.results = {'passed': 0, 'failed': 0, 'warnings': 0}
        self.report = []

    def _log_pass(self, message):
        self.report.append(f"{C_GREEN}✓ PASS:{C_END} {message}")
        self.results['passed'] += 1

    def _log_fail(self, message):
        self.report.append(f"{C_RED}✗ FAIL:{C_END} {message}")
        self.results['failed'] += 1

    def _log_warn(self, message):
        self.report.append(f"{C_YELLOW}⚠ WARN:{C_END} {message}")
        self.results['warnings'] += 1
        
    def _log_info(self, message):
        self.report.append(f"{C_BLUE}ℹ INFO:{C_END} {message}")

    async def run_verification(self):
        """
        The main method to run all verification steps.
        """
        print(f"{C_MAGENTA}{'='*70}{C_END}")
        print(f"{C_MAGENTA}  GRACE AI - COMPREHENSIVE SYSTEM CONNECTION & RUNTIME VERIFICATION{C_END}")
        print(f"{C_MAGENTA}{'='*70}{C_END}\n")

        # Step 1: Initialize the system
        print(f"{C_CYAN}--- STEP 1: Initializing System & Services ---{C_END}")
        try:
            # Use a mock argparse object
            class MockArgs:
                debug = False
                kernel = None
                dry_run = False
            
            self.launcher = GraceLauncher(MockArgs())
            await self.launcher.initialize()
            self._log_pass("System and Service Registry initialized successfully.")
        except Exception as e:
            self._log_fail(f"System initialization failed: {e}")
            self.print_summary()
            return
        print("Done.\n")

        # Step 2: Create Kernels
        print(f"{C_CYAN}--- STEP 2: Creating All Kernels ---{C_END}")
        try:
            await self.launcher.create_kernels()
            self._log_pass(f"Successfully created {len(self.launcher.kernels)} kernels.")
        except Exception as e:
            self._log_fail(f"Kernel creation failed: {e}")
            self.print_summary()
            return
        print("Done.\n")

        # Step 3: Verify Kernel-Service Wiring
        print(f"{C_CYAN}--- STEP 3: Verifying Kernel-Service Wiring ---{C_END}")
        if not self.launcher.kernels:
            self._log_fail("No kernels were created to verify.")
        else:
            for kernel in self.launcher.kernels:
                self._log_info(f"Verifying wiring for {C_YELLOW}{kernel.name}{C_END}...")
                try:
                    health = await kernel.health_check()
                    services = health.get('services', {})
                    all_wired = True
                    for service_name, status in services.items():
                        if status == 'missing':
                            self._log_fail(f"Kernel '{kernel.name}' is missing service: {C_RED}{service_name}{C_END}")
                            all_wired = False
                        else:
                            self._log_pass(f"Kernel '{kernel.name}' is wired to service: {C_GREEN}{service_name}{C_END}")
                    
                    if all_wired:
                        self._log_pass(f"All declared services for '{kernel.name}' are wired.")
                except Exception as e:
                    self._log_fail(f"Could not perform health check for kernel '{kernel.name}': {e}")
        print("Done.\n")

        # Step 4: Run Kernel Health Checks
        print(f"{C_CYAN}--- STEP 4: Running Kernel Health Checks ---{C_END}")
        if not self.launcher.kernels:
            self._log_fail("No kernels to run health checks on.")
        else:
            for kernel in self.launcher.kernels:
                try:
                    health = await kernel.health_check()
                    self._log_pass(f"Health check for '{kernel.name}': {health}")
                except Exception as e:
                    self._log_fail(f"Health check failed for kernel '{kernel.name}': {e}")
        print("Done.\n")
        
        # Step 5: Simulate a Task
        print(f"{C_CYAN}--- STEP 5: Simulating a Cognitive Task ---{C_END}")
        cortex = next((k for k in self.launcher.kernels if k.name == 'cognitive_cortex'), None)
        if cortex:
            try:
                task = {'type': 'plan', 'goal': 'Verify system integrity.'}
                result = await cortex.execute(task)
                if result.get('success'):
                    self._log_pass(f"Cognitive Cortex executed a test plan successfully. Tasks created: {result.get('task_ids')}")
                else:
                    self._log_fail(f"Cognitive Cortex task execution failed: {result.get('error')}")
            except Exception as e:
                self._log_fail(f"Exception during Cognitive Cortex task simulation: {e}")
        else:
            self._log_warn("Cognitive Cortex kernel not found, skipping task simulation.")
        print("Done.\n")

        self.print_summary()

    def print_summary(self):
        """
        Prints the final verification report.
        """
        print(f"\n{C_MAGENTA}{'='*70}{C_END}")
        print(f"{C_MAGENTA}                  VERIFICATION REPORT SUMMARY{C_END}")
        print(f"{C_MAGENTA}{'='*70}{C_END}\n")

        for line in self.report:
            print(line)

        print("\n--------------------------------------------------")
        print(f"  {C_GREEN}TOTAL PASSED: {self.results['passed']}{C_END}")
        print(f"  {C_RED}TOTAL FAILED: {self.results['failed']}{C_END}")
        print(f"  {C_YELLOW}TOTAL WARNINGS: {self.results['warnings']}{C_END}")
        print("--------------------------------------------------\n")

        if self.results['failed'] > 0:
            print(f"{C_RED}STATUS: ✗ System verification FAILED. Please review the errors above.{C_END}\n")
        else:
            print(f"{C_GREEN}STATUS: ✓ All systems are connected and running correctly!{C_END}\n")


if __name__ == "__main__":
    verifier = SystemVerifier()
    asyncio.run(verifier.run_verification())
