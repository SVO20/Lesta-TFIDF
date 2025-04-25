import sys
from functools import partial

from loguru import logger

# Initials
scenario = "ALL"    # "ALL" , "FILE_ONLY" , "STDOUTPUT_ONLY" accepted
level = "TRACE"  # "OMIT", "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR" ("CRITICAL")

enqueue = True

# Customization
logger.remove(None)

logger.level("OMIT", no=1, color="<light-black>", icon="_")  # Custom level for excluded log-events
if scenario in ("FILE_ONLY", "ALL"):
    logger.add("../logs/event_log.log",
               encoding="utf8",
               format="{time} | {level: <8} | {name: ^15} | {function: ^15} | {line: >3} | {message}",
               enqueue=enqueue,
               level=level,
               rotation="10 MB",        # New file every 10MB
               retention="30 days",     # Keep 30 lays
               compression="zip")       # Zip old logs
if scenario in ("STDOUTPUT_ONLY", "ALL"):
    logger.add(sys.stdout,
               format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                      "<level>{level: <8}</level> | "
                      "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
               enqueue=enqueue,
               level=level)

# Aliases
omit = partial(logger.log, "OMIT")
trace = logger.trace
debug = logger.debug
info = logger.info
success = logger.success
warning = logger.warning
error = logger.error
# for inline logging critical errors
err = lambda s: s if error("❌ " + s) else s
"""
# Template for use it all:
from logger import omit, trace, debug, info, success, warning, error, err
"""



