# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Launch an SGLang text classifier with multimodal processing disabled."""

import os
import sys

from sglang.launch_server import run_server
from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree


def main() -> None:
    """Parse normal SGLang arguments and force the text-only initialization
    path."""
    load_plugins()
    server_args = prepare_server_args(sys.argv[1:])
    server_args.enable_multimodal = False
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
