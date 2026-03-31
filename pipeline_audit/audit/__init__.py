"""Audit subpackage: reads only observable logs to detect hidden self-preserving behavior."""
from .group_discovery import PseudoLocusDiscovery
from .detector import AuditDetector
from .alarm import AlarmLogic

__all__ = ["PseudoLocusDiscovery", "AuditDetector", "AlarmLogic"]
