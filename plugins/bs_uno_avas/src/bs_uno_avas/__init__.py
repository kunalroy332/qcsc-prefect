"""Full-system broken-symmetry UHF + UNO-AVAS active spaces for Fe-S clusters.

Produces the artifact set + manifest (schema v1) read by the SBD flow through
``FlowParameters.active_space_manifest`` (``qcsc_workflow_utility.active_space``).
Orbital labels follow ``"<model idx>_<element>_<AO>"``, e.g. ``"0_Fe_3dxz"``, ``"2_S_3py"``.
"""

__version__ = "0.1"
