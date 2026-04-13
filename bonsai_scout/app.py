import sys

if sys.platform == "win32":
    import types
    m = types.ModuleType("resource")
    m.RLIMIT_AS = 0
    m.RUSAGE_SELF = 0
    m.getrusage = lambda x: types.SimpleNamespace(
        ru_maxrss=0, ru_utime=0, ru_stime=0
    )
    m.setrlimit = lambda x, y: None
    sys.modules["resource"] = m

from bonsai_scout_app import app