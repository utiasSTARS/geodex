"""autodoc support for the nanobind extension module of geodex.

Sphinx takes a nanobind method (``nb_method``) for an attribute descriptor, because it has
``__get__`` but none of the types Sphinx knows as routines, so a class documented with
``:members:`` shows its methods as attributes without signatures. A documenter that claims
nanobind routines inside a class documents them as methods, with the signature nanobind
writes as the first line of the docstring.

The bindings' docstrings are plain text, where ``|x|`` is a norm. reStructuredText reads it
as a substitution reference, so every bar in a docstring is escaped.
"""

from __future__ import annotations

from sphinx.ext.autodoc import ClassDocumenter, MethodDocumenter

_NANOBIND_ROUTINES = ("nb_method", "nb_func", "nb_bound_method")


class NanobindMethodDocumenter(MethodDocumenter):
    """A nanobind routine inside a class, documented as a method."""

    objtype = "nanobindmethod"
    directivetype = "method"
    priority = MethodDocumenter.priority + 20  # above the attribute documenter

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return (type(member).__name__ in _NANOBIND_ROUTINES
                and isinstance(parent, ClassDocumenter))


def _escape_bars(app, what, name, obj, options, lines):
    lines[:] = [line.replace("|", "\\|") for line in lines]


def setup(app):
    app.setup_extension("sphinx.ext.autodoc")
    app.add_autodocumenter(NanobindMethodDocumenter)
    app.connect("autodoc-process-docstring", _escape_bars)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
