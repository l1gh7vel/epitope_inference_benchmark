"""The WALLE checkpoint, used by every task.

AsEP release ten checkpoints and recommend none. This project uses
`treasured-sweep-8` everywhere. It is the checkpoint the AsEP inference README
designates as best on validation bipartite-edge MCC under their epitope-ratio
split. That split is the easier of their two settings, and edge MCC is not the
node-level metric our per-residue AUPRC matches. Both limits are stated in the
Methods.
"""

HEADLINE_CHECKPOINT = "treasured-sweep-8"

#: Structure source the WALLE scores were produced from, per task. Task I
#: reads the deposited structures; Tasks II and III read AlphaFold3's.
SOURCE = {"task1": "experimental", "task2": "af3", "task3": "af3", "task3cov": "af3"}
