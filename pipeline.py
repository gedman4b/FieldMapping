"""
pipeline.py

Schema Field Mapper Pipeline.

Maps every field in a source schema to its semantically equivalent field in a
destination schema, using an LLM at two stages, respecting the constraint that
no single LLM prompt receives both full schemas.

Architecture (four stages):

  [1] Schema parsing (deterministic, no LLM)
      Parse both schemas into typed Pydantic models. Flatten MongoDB nested
      paths to dot notation.

  [2] Table matching (LLM, one call)
      Send only table/collection names and one-line purposes. Get back pairings.

  [3] Field mapping (LLM, one call per matched table pair)
      For each pair, send only that pair's fields. Schema-constrained JSON.

  [4] Enrichment and verification (deterministic)
      Apply deterministic rules the LLM tends to miss (for example, MongoDB
      unique-index reminders for source fields with UNIQUE constraints).
      Cross-check completeness, validity, duplicate detection. Assemble output.

Design principles:

  - No LLM call sees both full schemas. Constraint respected by construction.
  - Every LLM call has schema-constrained output. LLM outputs feed code, not
    users.
  - Deterministic stages sandwich the LLM stages. Verification catches missing
    fields, duplicates, invalid destination paths before output.
  - The pipeline scales: at 1000 source tables you make 1 table-match call
    plus 1000 field-map calls (parallelizable), not one impossible mega-prompt.

Author: Scott Josephson
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# =============================================================================
# Pydantic models: inter-stage contracts
# =============================================================================

class SourceField(BaseModel):
    """A single field in a source (MySQL) table."""
    name: str
    type: str                                          # e.g. "VARCHAR(50)"
    constraints: List[str] = Field(default_factory=list)
    comment: Optional[str] = None


class SourceTable(BaseModel):
    """A source table with its fields and a one-line purpose hint."""
    name: str
    purpose_hint: Optional[str] = None
    fields: List[SourceField]


class DestField(BaseModel):
    """A single destination path (MongoDB), nested paths pre-flattened."""
    path: str                                          # e.g. "fullName.firstName"
    type: str                                          # e.g. "String", "ObjectId"
    comment: Optional[str] = None


class DestCollection(BaseModel):
    """A destination collection with its paths and a one-line purpose hint."""
    name: str
    purpose_hint: Optional[str] = None
    fields: List[DestField]


class FieldMapping(BaseModel):
    """A single field-level mapping. destination_field may be None if
    the source field has no destination counterpart."""
    source_field: str
    destination_field: Optional[str]
    type_transform: str
    confidence: float
    reasoning: str
    notes: Optional[str] = None


class TableMapping(BaseModel):
    """The mapping for one source-table / destination-collection pair."""
    source_table: str
    destination_collection: str
    confidence: float
    reasoning: str
    field_mappings: List[FieldMapping]
    unmapped_source_fields: List[str] = Field(default_factory=list)
    unmapped_destination_fields: List[str] = Field(default_factory=list)


class MappingOutput(BaseModel):
    """Top-level pipeline output."""
    mapping_version: str = "1.0"
    source: str
    destination: str
    generated_at: str
    tables: List[TableMapping]


# =============================================================================
# Stage 1: Schema parsing (deterministic)
# =============================================================================

def parse_source_schema() -> List[SourceTable]:
    """Parse the MySQL source schema.

    In production this would parse a schema file, DDL, or query information_schema.
    For this take-home the schema is embedded as it was provided.
    """
    return [
        SourceTable(
            name="emp_master",
            purpose_hint="employee master records",
            fields=[
                SourceField(name="emp_id", type="INT", constraints=["PRIMARY KEY"]),
                SourceField(name="emp_cd", type="VARCHAR(20)",
                            constraints=["UNIQUE", "NOT NULL"],
                            comment="human-readable employee code"),
                SourceField(name="f_name", type="VARCHAR(50)", constraints=["NOT NULL"]),
                SourceField(name="l_name", type="VARCHAR(50)", constraints=["NOT NULL"]),
                SourceField(name="dob", type="DATE"),
                SourceField(name="hire_dt", type="DATETIME"),
                SourceField(name="term_dt", type="DATETIME",
                            comment="null if still active"),
                SourceField(name="dept_id", type="INT",
                            constraints=["FK -> dept_info.dept_id"]),
                SourceField(name="mgr_emp_id", type="INT",
                            constraints=["FK -> emp_master.emp_id"]),
                SourceField(name="job_lvl_cd", type="VARCHAR(10)",
                            comment="e.g. L1, L2, IC3, M1"),
                SourceField(name="base_sal", type="DECIMAL(12,2)"),
                SourceField(name="sal_currency", type="CHAR(3)",
                            comment="ISO 4217, e.g. USD"),
                SourceField(name="work_email", type="VARCHAR(120)",
                            constraints=["UNIQUE"]),
                SourceField(name="work_phone", type="VARCHAR(20)"),
                SourceField(name="office_loc_id", type="INT",
                            constraints=["FK -> locations.loc_id"]),
                SourceField(name="is_remote", type="TINYINT(1)",
                            comment="0 or 1"),
                SourceField(name="rec_stat", type="CHAR(1)",
                            comment="A=Active, I=Inactive, T=Terminated"),
                SourceField(name="created_ts", type="DATETIME",
                            comment="record creation timestamp"),
                SourceField(name="updated_ts", type="DATETIME",
                            comment="last update timestamp"),
            ],
        ),
        SourceTable(
            name="dept_info",
            purpose_hint="department records",
            fields=[
                SourceField(name="dept_id", type="INT", constraints=["PRIMARY KEY"]),
                SourceField(name="dept_cd", type="VARCHAR(20)", constraints=["UNIQUE"]),
                SourceField(name="dept_nm", type="VARCHAR(100)"),
                SourceField(name="parent_dept_id", type="INT",
                            constraints=["FK -> dept_info.dept_id"],
                            comment="self-referencing"),
                SourceField(name="dept_head_id", type="INT",
                            constraints=["FK -> emp_master.emp_id"]),
                SourceField(name="cost_ctr_cd", type="VARCHAR(20)",
                            comment="finance cost center code"),
                SourceField(name="dept_stat", type="CHAR(1)",
                            comment="A=Active, I=Inactive"),
            ],
        ),
        SourceTable(
            name="locations",
            purpose_hint="physical office and facility locations",
            fields=[
                SourceField(name="loc_id", type="INT", constraints=["PRIMARY KEY"]),
                SourceField(name="loc_cd", type="VARCHAR(20)", constraints=["UNIQUE"]),
                SourceField(name="loc_nm", type="VARCHAR(100)"),
                SourceField(name="city", type="VARCHAR(80)"),
                SourceField(name="state_prov", type="VARCHAR(80)"),
                SourceField(name="country_cd", type="CHAR(2)",
                            comment="ISO 3166-1 alpha-2"),
                SourceField(name="postal_cd", type="VARCHAR(20)"),
                SourceField(name="tz_cd", type="VARCHAR(50)", comment="IANA timezone"),
            ],
        ),
    ]


def parse_destination_schema() -> List[DestCollection]:
    """Parse the MongoDB destination schema with nested paths flattened."""
    return [
        DestCollection(
            name="employees",
            purpose_hint="employee documents",
            fields=[
                DestField(path="_id", type="ObjectId"),
                DestField(path="employeeCode", type="String",
                          comment="unique human-readable ID"),
                DestField(path="fullName.firstName", type="String"),
                DestField(path="fullName.lastName", type="String"),
                DestField(path="employment.startDate", type="ISODate"),
                DestField(path="employment.endDate", type="ISODate",
                          comment="null if currently employed"),
                DestField(path="employment.status", type="String",
                          comment="active / inactive / terminated"),
                DestField(path="employment.jobLevel", type="String",
                          comment="e.g. L1, IC3, M1"),
                DestField(path="employment.isRemote", type="Boolean"),
                DestField(path="employment.managerId", type="ObjectId",
                          comment="ref -> employees._id"),
                DestField(path="compensation.baseSalary", type="Number"),
                DestField(path="compensation.currency", type="String",
                          comment="ISO 4217"),
                DestField(path="contact.email", type="String"),
                DestField(path="contact.phone", type="String"),
                DestField(path="department.departmentId", type="ObjectId",
                          comment="ref -> departments._id"),
                DestField(path="department.code", type="String"),
                DestField(path="department.name", type="String"),
                DestField(path="location.locationId", type="ObjectId",
                          comment="ref -> locations._id"),
                DestField(path="location.code", type="String"),
                DestField(path="location.name", type="String"),
                DestField(path="location.city", type="String"),
                DestField(path="location.country", type="String",
                          comment="ISO 3166-1 alpha-2"),
                DestField(path="location.timezone", type="String",
                          comment="IANA timezone"),
                DestField(path="meta.createdAt", type="ISODate"),
                DestField(path="meta.updatedAt", type="ISODate"),
            ],
        ),
        DestCollection(
            name="departments",
            purpose_hint="department documents",
            fields=[
                DestField(path="_id", type="ObjectId"),
                DestField(path="code", type="String"),
                DestField(path="name", type="String"),
                DestField(path="parentDepartmentId", type="ObjectId",
                          comment="self-ref"),
                DestField(path="headEmployeeId", type="ObjectId",
                          comment="ref -> employees._id"),
                DestField(path="costCenterCode", type="String"),
                DestField(path="isActive", type="Boolean"),
            ],
        ),
        DestCollection(
            name="locations",
            purpose_hint="location documents",
            fields=[
                DestField(path="_id", type="ObjectId"),
                DestField(path="code", type="String"),
                DestField(path="name", type="String"),
                DestField(path="city", type="String"),
                DestField(path="stateOrProvince", type="String"),
                DestField(path="country", type="String",
                          comment="ISO 3166-1 alpha-2"),
                DestField(path="postalCode", type="String"),
                DestField(path="timezone", type="String", comment="IANA timezone"),
            ],
        ),
    ]


# =============================================================================
# LLM client (Anthropic Claude)
# =============================================================================

class LLMClient:
    """Thin wrapper. Swap for any provider (OpenAI, local, etc.) by
    re-implementing .call()."""

    def __init__(self, model: str = "claude-sonnet-4-5"):
        self.client = anthropic.Anthropic()
        self.model = model

    def call(self, system: str, user: str, max_tokens: int = 4000) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


# =============================================================================
# Stage 2: Table matcher (LLM, one call for ALL table pairs)
# =============================================================================

TABLE_MATCH_SYSTEM = """You match tables between two database schemas. You will be
given only table/collection NAMES and one-line purposes. You will NOT be given
their field lists. Propose which source table maps to which destination collection.

Return only a JSON object in this exact form:
{"pairs": [
  {"source_table": "...", "destination_collection": "..." or null,
   "confidence": 0.0, "reasoning": "..."}
]}

Every source table must appear. If no destination collection matches a source table,
set destination_collection to null. Confidence is 0.0 to 1.0. Reasoning is one
sentence."""


def match_tables(
    client: LLMClient,
    sources: List[SourceTable],
    dests: List[DestCollection],
) -> Dict[str, Any]:
    """Stage 2. Send only names and one-line purposes. Never send fields."""
    lines = ["Source tables (names and purposes only, NO field lists):"]
    for t in sources:
        lines.append(f"  - {t.name}: {t.purpose_hint or '(no hint provided)'}")
    lines.append("")
    lines.append("Destination collections (names and purposes only, NO field lists):")
    for d in dests:
        lines.append(f"  - {d.name}: {d.purpose_hint or '(no hint provided)'}")
    lines.append("")
    lines.append("Return the JSON.")

    prompt = "\n".join(lines)
    raw = client.call(TABLE_MATCH_SYSTEM, prompt, max_tokens=1000)
    return _extract_json(raw)


# =============================================================================
# Stage 3: Field mapper (LLM, one call per matched pair)
# =============================================================================

FIELD_MAP_SYSTEM = """You map fields between one source table and one destination
collection. You are given ONLY those two entities' fields. You are NOT given any
other tables or collections.

For EVERY source field, propose the best-matching destination path, or set
destination_field to null if no reasonable match exists.

Also list destination paths NOT filled by any source field in this table
(unmapped_destination_fields). Those paths may be populated during migration
via joins from other tables; that is expected.

Common type transformations:
- INT PRIMARY KEY -> ObjectId (needs ID generation strategy)
- INT FK -> ObjectId (needs FK translation via an ID mapping table)
- TINYINT(1) -> Boolean
- CHAR(1) enum codes -> String enum (value transform required)
- DATE / DATETIME -> ISODate (UTC normalization)
- DECIMAL -> Number (consider Decimal128 if exact precision matters)
- CHAR(3) / CHAR(2) / VARCHAR -> String (usually direct)

Confidence calibration (this matters):
- Reserve 1.0 for mappings you would stake your reputation on. In practice,
  mappings that require a type transform, a value transform, an FK
  translation, or an ID generation strategy should NOT be 1.0. Something
  can still go wrong operationally even when the mapping is obviously right.
- Use 0.95 to 0.99 for high-confidence mappings with mechanical transforms
  (INT PK -> ObjectId, TINYINT(1) -> Boolean, VARCHAR -> String, ISO code
  fields with matching standards on both sides).
- Use 0.85 to 0.94 for mappings that involve semantic judgment (a source
  field matched to one of several plausible destinations, enum-code to
  String enum with a value transform, or a name-based match where the
  underlying semantics are inferred rather than declared).
- Use below 0.85 when there is genuine uncertainty (ambiguous label
  matches, multiple plausible destinations, or missing metadata).
- Do NOT default to 1.0 across every field. Downstream systems use
  confidence to route mappings to human review; a uniform 1.0 makes that
  routing impossible and defeats the point of surfacing confidence at all.
- If you catch yourself about to write 1.0 more than three times in a
  single table pair, stop and re-rank: some of those are almost certainly
  in the 0.95 to 0.99 band, not at the ceiling.

Return only a JSON object in this exact form:
{
  "confidence": 0.0,
  "reasoning": "one sentence about the table-pair match",
  "field_mappings": [
    {"source_field": "...", "destination_field": "..." or null,
     "type_transform": "...", "confidence": 0.0,
     "reasoning": "...", "notes": "..." or null}
  ],
  "unmapped_source_fields": ["..."],
  "unmapped_destination_fields": ["..."]
}

Every source field MUST appear as an item in field_mappings. Source fields with
no destination must also be listed in unmapped_source_fields for quick lookup."""


def map_fields(
    client: LLMClient,
    source_table: SourceTable,
    dest_collection: DestCollection,
) -> Dict[str, Any]:
    """Stage 3. Send ONE source table's fields plus ONE destination collection's
    paths. Never send other tables or collections."""
    lines = [f"Source table: {source_table.name} (MySQL)"]
    lines.append("Source fields:")
    for f in source_table.fields:
        line = f"  - {f.name}: {f.type}"
        if f.constraints:
            line += f" [{', '.join(f.constraints)}]"
        if f.comment:
            line += f"  -- {f.comment}"
        lines.append(line)
    lines.append("")

    lines.append(f"Destination collection: {dest_collection.name} (MongoDB)")
    lines.append("Destination paths (nested paths flattened to dot notation):")
    for d in dest_collection.fields:
        line = f"  - {d.path}: {d.type}"
        if d.comment:
            line += f"  -- {d.comment}"
        lines.append(line)
    lines.append("")
    lines.append("Return the JSON.")

    prompt = "\n".join(lines)
    raw = client.call(FIELD_MAP_SYSTEM, prompt, max_tokens=4000)
    return _extract_json(raw)


# =============================================================================
# Stage 4: Enrichment and Verification (deterministic)
# =============================================================================
#
# Enrichment applies deterministic rules the LLM often misses. Verification
# then checks structural correctness. Both run per table pair; both are cheap
# and reproducible.


def enrich_mapping(
    source_table: SourceTable,
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply deterministic rules to a proposed field mapping in place.

    Rationale
    ---------
    LLM mapping proposals are semantically strong but operationally patchy.
    A production pipeline should not depend on the LLM catching every
    infrastructure detail. Rules that can be applied by inspecting the
    parsed source schema go here instead, where they are cheap, reliable,
    testable, and consistent across runs.

    Rules currently applied
    -----------------------
    - UNIQUE constraint preservation:
      MongoDB does NOT automatically preserve MySQL uniqueness. If a source
      field carries the UNIQUE constraint AND maps to a non-null, non-_id
      destination path, append a note reminding the migration engineer to
      create a corresponding unique index in MongoDB on that path. The
      _id case is excluded because MongoDB _id is inherently unique.

    Rules to add as they emerge from real migrations:
    - NOT NULL preservation (add MongoDB schema validation rule)
    - CHECK constraints (translate to MongoDB $jsonSchema validators)
    - DEFAULT values (surface as migration-time default population)
    - CHAR-length limits (surface as MongoDB validation)

    Deterministic rules are cheaper, more reliable, and more testable than
    prompting the LLM to catch them. Add rules here rather than expanding
    the prompt.

    Returns the mapping dict, mutated in place for convenience.
    """
    src_by_name = {f.name: f for f in source_table.fields}

    for m in mapping.get("field_mappings", []):
        src_name = m.get("source_field")
        src = src_by_name.get(src_name)
        if src is None:
            # verify_mapping will flag this as an unknown source field
            continue

        dest = m.get("destination_field")

        # ---- UNIQUE constraint rule ----
        has_unique = any("UNIQUE" in c.upper() for c in src.constraints)
        if has_unique and dest and dest != "_id":
            unique_note = (
                f"Source field carries a UNIQUE constraint in MySQL. "
                f"Create a corresponding unique index in MongoDB on "
                f"'{dest}' to preserve the uniqueness invariant, since "
                f"MongoDB does not carry the constraint over automatically."
            )
            existing = m.get("notes")
            if existing:
                # If the LLM already mentioned unique-index guidance, do
                # not duplicate. Otherwise, append.
                if "unique index" not in existing.lower():
                    m["notes"] = f"{existing.rstrip('.')} . {unique_note}"
            else:
                m["notes"] = unique_note

    return mapping


def verify_mapping(
    source_table: SourceTable,
    dest_collection: DestCollection,
    mapping: Dict[str, Any],
) -> List[str]:
    """Deterministic checks over the LLM's field mapping. Returns warnings.
    Empty list means clean. Non-empty warnings are surfaced but do not block
    output; the caller decides whether to re-prompt or accept."""
    warnings: List[str] = []

    mapped_source_names = {m["source_field"] for m in mapping["field_mappings"]}
    all_source_names = {f.name for f in source_table.fields}

    missing = all_source_names - mapped_source_names
    if missing:
        warnings.append(f"missing source fields: {sorted(missing)}")

    extra = mapped_source_names - all_source_names
    if extra:
        warnings.append(f"unknown source fields in output: {sorted(extra)}")

    all_dest_paths = {d.path for d in dest_collection.fields}
    for m in mapping["field_mappings"]:
        dest = m.get("destination_field")
        if dest is not None and dest not in all_dest_paths:
            warnings.append(f"unknown destination path: {dest}")

    for m in mapping["field_mappings"]:
        c = m.get("confidence", 0)
        if not (0.0 <= c <= 1.0):
            warnings.append(f"confidence out of range for {m['source_field']}: {c}")

    # Check every non-null destination is used at most once
    dest_uses: Dict[str, List[str]] = {}
    for m in mapping["field_mappings"]:
        d = m.get("destination_field")
        if d:
            dest_uses.setdefault(d, []).append(m["source_field"])
    for d, sources_for_d in dest_uses.items():
        if len(sources_for_d) > 1:
            warnings.append(
                f"destination path {d} claimed by multiple sources: {sources_for_d}"
            )

    return warnings


# =============================================================================
# Orchestrator
# =============================================================================

def run_pipeline(output_path: Path) -> MappingOutput:
    """Run all four stages end to end."""
    log.info("stage 1: parsing schemas (deterministic)")
    sources = parse_source_schema()
    dests = parse_destination_schema()
    log.info("  parsed %d source tables, %d destination collections",
             len(sources), len(dests))

    log.info("stage 2: matching tables to collections (LLM, one call)")
    client = LLMClient()
    table_pairs = match_tables(client, sources, dests)
    log.info("  proposed %d table pairs", len(table_pairs["pairs"]))

    source_by_name = {t.name: t for t in sources}
    dest_by_name = {d.name: d for d in dests}

    table_mappings: List[TableMapping] = []
    for pair in table_pairs["pairs"]:
        st = pair["source_table"]
        dc = pair["destination_collection"]

        if dc is None:
            # No destination match; record whole source as unmapped
            src = source_by_name[st]
            table_mappings.append(TableMapping(
                source_table=st,
                destination_collection="(unmapped)",
                confidence=pair["confidence"],
                reasoning=pair["reasoning"],
                field_mappings=[],
                unmapped_source_fields=[f.name for f in src.fields],
                unmapped_destination_fields=[],
            ))
            continue

        log.info("stage 3: mapping fields for %s -> %s (LLM, one call)", st, dc)
        src = source_by_name[st]
        dest = dest_by_name[dc]
        mapping = map_fields(client, src, dest)

        log.info("stage 4a: enriching mapping for %s -> %s (deterministic rules)",
                 st, dc)
        mapping = enrich_mapping(src, mapping)

        log.info("stage 4b: verifying mapping for %s -> %s (deterministic checks)",
                 st, dc)
        warnings = verify_mapping(src, dest, mapping)
        for w in warnings:
            log.warning("  %s -> %s: %s", st, dc, w)

        table_mappings.append(TableMapping(
            source_table=st,
            destination_collection=dc,
            confidence=pair["confidence"],
            reasoning=pair["reasoning"],
            field_mappings=[FieldMapping(**m) for m in mapping["field_mappings"]],
            unmapped_source_fields=mapping.get("unmapped_source_fields", []),
            unmapped_destination_fields=mapping.get("unmapped_destination_fields", []),
        ))

    output = MappingOutput(
        source="legacy_hrm (MySQL)",
        destination="people_platform (MongoDB)",
        generated_at=datetime.now(timezone.utc).isoformat(),
        tables=table_mappings,
    )

    output_path.write_text(json.dumps(output.model_dump(), indent=2))
    log.info("done. wrote mapping to %s", output_path)
    return output


# =============================================================================
# Helpers
# =============================================================================

def _extract_json(raw: str) -> Dict[str, Any]:
    """Extract the first JSON object from a raw LLM response."""
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"no JSON object in LLM output:\n{raw[:500]}")
    return json.loads(raw[start:end])


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Schema field mapper pipeline. "
                    "Maps MySQL fields to MongoDB paths via a two-stage LLM pipeline."
    )
    ap.add_argument("--out", default="mapping.json",
                    help="Output JSON path (default: mapping.json)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY environment variable not set")
        raise SystemExit(1)

    run_pipeline(Path(args.out))
