"""Which messages ask for the FULL picture (evidence, sources, why, which tools, full report). Shared by the router (a question about the last answer is a follow-up, never a re-run)
and the writer (long report instead of the brief)."""
import re

DETAIL_ASK = re.compile(r"\b(evidence|proof|prove|sources?|which (tools?|functions?|data|datasets?|numbers)|what (tools?|functions?|data|datasets?) (did|do|was|were)|"
                        r"(tools?|functions?) (you|did you) (call|use)\w*|how did you (calculate|compute|decide|get|work|arrive|know|come)|how do you know|"
                        r"show (me )?(the )?(details?|steps?|workings?|calculations?|numbers|data|reasoning|method)|(more|full|complete|detailed) (details?|report|explanation|picture|answer)|"
                        r"in detail|methodology|reasoning|audit|trace|full report)\b", re.I)
ARGUE = re.compile(r"\bwhy\b|explain (that|why|how|this|it|your|the)|can you explain|please explain|justify|argument|how come|reason for|how do you know|based on what|on what basis|what makes you", re.I)
