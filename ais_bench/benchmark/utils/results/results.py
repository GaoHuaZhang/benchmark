import json
import fcntl


def safe_write(results_dict: dict, filename):
    """
    use fcntl file lock to implement mutual exclusion writing
    """
    with open(filename, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            results=[json.dumps(result, ensure_ascii=False) + "\n" for result in results_dict.values()]
            f.writelines(results)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
