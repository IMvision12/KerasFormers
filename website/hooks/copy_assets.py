"""Make the repo-root ``assets/`` directory part of the MkDocs build.

The model pages carry their figures and audio as raw HTML pointing at
``../assets/...``, but that directory lives next to ``docs/`` rather than inside
it, so MkDocs has no idea it exists and every image and audio player 404s.

Registering the files here (rather than copying them afterwards) means the same
thing happens for ``mkdocs build`` and ``mkdocs serve``, and MkDocs handles the
copying itself.

Pages are served from their own directory (``/whisper/``), so ``../assets/x.jpg``
resolves to ``/assets/x.jpg``: these land at exactly that path, alongside the
theme's own ``assets/javascripts`` and ``assets/stylesheets``.
"""

import os

from mkdocs.structure.files import File

ASSETS_DIR = "assets"


def find_media_root(start):
    """Walk up from ``start`` to the directory that holds ``assets/``.

    This config lives in ``website/`` while ``assets/`` sits at the repo root, so
    the two are not siblings; searching upwards keeps the hook working wherever
    the config is moved to.
    """
    current = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(current, ASSETS_DIR)):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def on_files(files, config):
    config_dir = os.path.dirname(os.path.abspath(config["config_file_path"]))

    # Theme overrides (the stylesheet in extra_css) live beside this hook rather
    # than in docs/, so register them the same way as the media.
    overrides = os.path.join(config_dir, "overrides")
    for dirpath, _, filenames in os.walk(overrides):
        for filename in filenames:
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, overrides).replace(os.sep, "/")
            files.append(
                File(
                    rel_path,
                    src_dir=overrides,
                    dest_dir=config["site_dir"],
                    use_directory_urls=False,
                )
            )

    root = find_media_root(config_dir)
    if root is None:
        return files
    assets_root = os.path.join(root, ASSETS_DIR)

    for dirpath, _, filenames in os.walk(assets_root):
        for filename in filenames:
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, root).replace(os.sep, "/")
            files.append(
                File(
                    rel_path,
                    src_dir=root,
                    dest_dir=config["site_dir"],
                    use_directory_urls=False,
                )
            )
    return files
