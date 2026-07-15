*Schema Field Mapper*

Design, Implementation, and Reasoning

AI / LLM Engineer  |  Prepared by Scott Josephson

**Executive summary**

I built a four-stage pipeline. Two deterministic stages sandwich two LLM stages. The LLM stages are structured so that no single call ever sees both full schemas: the first LLM call sees only table names and one-line purposes to identify which source table pairs with which destination collection; the second LLM call runs once per matched pair and sees only that pair's fields. Verification is deterministic and catches missing fields, invalid destination paths, duplicate destination claims, and confidence out-of-range issues before output.

The pipeline scales cleanly. A schema with N source tables and M destination collections requires one table-matching call plus N field-mapping calls (parallelizable), not one impossible mega-prompt. At 1000 tables the field-mapping stage runs in parallel with bounded context per call.

The deliverables in this submission are: pipeline.py containing the full pipeline implementation; mapping.json containing the generated output; and this write-up explaining prompt structure and design decisions.