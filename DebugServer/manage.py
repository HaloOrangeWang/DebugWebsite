#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from SearchEngine import SearchEng


def main():
    """Run administrative tasks."""

    # 初始化搜索算法。只在运行服务的时候初始化
    if sys.argv[1].lower() == 'runserver':
        SearchEng.preparation()

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'DebugServer.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
