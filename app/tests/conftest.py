"""Test-only shim for an AppTest bug in older Streamlit (seen in 1.51, fixed upstream by 1.67):
re-running a page that has a single-select st.segmented_control raised
"TypeError: 'int' object is not iterable", because ButtonGroup.indices iterated a single value.
The app itself is not affected; this only lets AppTest re-run such pages on the older version."""
from streamlit.testing.v1 import element_tree as _et

_BG = getattr(_et, "ButtonGroup", None)
if _BG is not None and not hasattr(_BG, "formatted_values"):          # the fixed versions have this property

    def _indices(self):
        v = self.value
        values = v if isinstance(v, (list, tuple)) else ([] if v is None else [v])
        return [self.options.index(self.format_func(x)) for x in values]

    _BG.indices = property(_indices)
