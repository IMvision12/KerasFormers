"""Fail the docs build if any image or audio reference in the built site is dead.

The model pages carry figures and audio as raw HTML (``<img src="../assets/...">``,
``<audio src="../assets/...">``) pointing at the repo-root ``assets/`` directory,
which MkDocs does not know about. The workflow copies that directory into the
built site; this script proves every reference actually landed somewhere.
"""

import pathlib
import posixpath
import re
import sys

# website/scripts/ -> repo root -> site/. Override by passing a path.
DEFAULT_SITE = pathlib.Path(__file__).resolve().parents[2] / "site"
PATTERN = re.compile(r'src="((?:\.\./|\./)?assets/[^"]+)"')


def main():
    SITE = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SITE
    if not SITE.is_dir():
        print(f"no built site at {SITE}", file=sys.stderr)
        return 1

    checked, bad = 0, []
    for html in SITE.rglob("*.html"):
        rel = html.parent.relative_to(SITE).as_posix()
        page = "/" if rel == "." else f"/{rel}/"
        for match in PATTERN.finditer(
            html.read_text(encoding="utf-8", errors="ignore")
        ):
            ref = match.group(1)
            target = posixpath.normpath(posixpath.join(page, ref)).lstrip("/")
            checked += 1
            if not (SITE / target).exists():
                bad.append(f"{page} -> {ref} (resolved to {target})")

    print(f"checked {checked} media references")
    if bad:
        print(f"{len(bad)} broken:", file=sys.stderr)
        for line in bad:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("all resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
