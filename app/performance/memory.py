# Re-export from app.core.performance.memory for backward compatibility
from app.core.performance.memory import track_memory  # noqa: F401
from app.core.performance.memory import get_memory_stats  # noqa: F401
from app.core.performance.memory import force_garbage_collection  # noqa: F401
