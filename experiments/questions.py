"""The two experiment questions and the protocol.

Q1  the workbook's disruption question (TRAINING #3): answerable, with ground truth recomputable from the raw csv.
Q2  a FOLLOW-UP that only makes sense after Q1 (same session). It exists to make the effect of memory visible.
Q1r Q1 asked again in a NEW session. It exists to expose cross-session memory (episodic) and run-to-run consistency.

Turn order in every arm: Q1 -> Q2 (same session) -> Q1r (fresh session).
"""
Q1 = ("Line U6 is suspended on a section between Hallesches Tor and Kaiserin-Augusta-Strasse stations. What is the reason "
      "behind this closure and how long will it last? How should the passengers be rerouted, which stations would become "
      "overloaded, and where should additional staff be deployed?")
Q2 = "Which of those stations should get staff first, and how sure are you about that ranking?"

PROTOCOL = [("Q1", Q1, "new"), ("Q2", Q2, "same"), ("Q1r", Q1, "new")]   # (turn id, text, session: new | same as previous)
