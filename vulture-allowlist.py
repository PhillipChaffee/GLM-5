"""vulture allowlist: references that mark dynamically-used names as used.

One reference per line, each with a reason comment. Empty by design: add an
entry only when vulture reports a name the code uses through dynamic access,
framework injection, or a name table. The file is scanned together with the
package (`vulture scripts vulture-allowlist.py`), so a reference here
counts as a use.
"""
