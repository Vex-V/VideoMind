import hashlib
import uuid
import warnings

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from ..paths import VECTOR_DIR, ensure as ensure_dirs
from .embedder import DEFAULT_MODEL, get_embedder
from .render import SUBJECT_VECTOR_FIELDS, VECTOR_FIELDS

COLLECTION = "chunks"

# One point per person, alongside the one-point-per-chunk collection. Kept
# separate rather than mixed in because the two answer different questions and
# a shared collection would make every chunk search filter subjects out.
SUBJECTS = "subjects"

# Every point, whichever collection, is located the same way: which video, which
# analyzer, which chunking, when. Defined once so the two collections cannot
# drift apart on the fields they share.
LOCATOR_FIELDS = {
    "video_id": "keyword",
    "extractor_id": "keyword",
    # Lets a caller that already knows which chunks it wants search exactly
    # those, instead of re-deriving them from a time range.
    "chunk_id": "integer",
    "chunk_config": "keyword",
    "start": "float",
    "end": "float",
}

# Payload fields worth an index: everything queries filter on.
INDEXED_FIELDS = {
    **LOCATOR_FIELDS,
    "objects": "keyword",
    "tags": "keyword",
    "people": "keyword",
    "speakers": "keyword",
    "setting": "text",
    # Counting is what embeddings cannot do: "mostly empty store" retrieved the
    # busiest chunk in the video, because a description listing five people
    # reads the same to a vector as one listing two.
    "people_count": "integer",
}

SUBJECT_INDEXED_FIELDS = LOCATOR_FIELDS


# The one place a filter is defined. Callers pass a flat dict of these names,
# so adding a filter here makes it reachable from the API immediately - the
# previous hand-written parameter list meant six store filters were silently
# unreachable because nobody had plumbed them through.
#   kind: any / not_any -> payload field matches (or avoids) one of the values
#         exact         -> payload field equals the value
#         gte / lte     -> numeric bound on `key`
LOCATOR_FILTERS: dict[str, dict] = {
    "video_ids":     {"key": "video_id",     "kind": "any",   "type": "list[str]"},
    "chunk_ids":     {"key": "chunk_id",     "kind": "any",   "type": "list[int]"},
    "analyzer_ids":  {"key": "extractor_id", "kind": "any",   "type": "list[str]"},
    "chunk_config":  {"key": "chunk_config", "kind": "exact", "type": "str"},
    "after":         {"key": "start",        "kind": "gte",   "type": "float"},
    "before":        {"key": "end",          "kind": "lte",   "type": "float"},
}

FILTER_SPEC: dict[str, dict] = {
    **LOCATOR_FILTERS,
    "objects":       {"key": "objects",      "kind": "any",   "type": "list[str]"},
    "tags":          {"key": "tags",         "kind": "any",   "type": "list[str]"},
    "speakers":      {"key": "speakers",     "kind": "any",   "type": "list[str]"},
    "people":        {"key": "people",       "kind": "any",   "type": "list[str]"},
    "min_people":    {"key": "people_count", "kind": "gte",   "type": "int"},
    "max_people":    {"key": "people_count", "kind": "lte",   "type": "int"},
}

# Subjects get the locators plus one of their own. They deliberately do *not*
# get the chunk-shaped keys (tags, speakers, people_count), which would validate
# and then silently match nothing.
#
# `exclude_video_ids` is what makes cross-video work expressible - "people who
# look like this one, in any video except their own" is a nearest-neighbour
# search, and without a negative filter it would return the source person's own
# sightings as its best matches.
SUBJECT_FILTER_SPEC: dict[str, dict] = {
    **LOCATOR_FILTERS,
    "exclude_video_ids": {"key": "video_id", "kind": "not_any", "type": "list[str]"},
}


def _suggest(name: str, spec: dict) -> str:
    """Nearest known filter name, so a typo says what was meant."""
    import difflib

    close = difflib.get_close_matches(name, spec, n=1, cutoff=0.6)
    return f"; did you mean {close[0]!r}?" if close else ""


def _conditions(filters: dict, spec: dict) -> tuple[list[FieldCondition], list[FieldCondition]]:
    """Turn a flat filter dict into (must, must_not), rejecting unknown keys.

    Validation is the point: a silently ignored filter returns plausible but
    wrong results, which is far worse than an error.
    """
    must: list[FieldCondition] = []
    must_not: list[FieldCondition] = []
    for name, value in filters.items():
        if value is None or value == []:
            continue
        entry = spec.get(name)
        if entry is None:
            raise ValueError(
                f"Unknown filter {name!r}{_suggest(name, spec)} "
                f"(known: {sorted(spec)})"
            )
        key, kind = entry["key"], entry["kind"]
        if kind in ("any", "not_any"):
            values = value if isinstance(value, (list, tuple, set)) else [value]
            condition = FieldCondition(key=key, match=MatchAny(any=list(values)))
            (must_not if kind == "not_any" else must).append(condition)
        elif kind == "exact":
            must.append(FieldCondition(key=key, match=MatchValue(value=value)))
        elif kind == "gte":
            must.append(FieldCondition(key=key, range=Range(gte=value)))
        elif kind == "lte":
            must.append(FieldCondition(key=key, range=Range(lte=value)))
    return must, must_not


def point_id(
    video_id: str, extractor_id: str, start: float, end: float,
    chunk_config: str, index: int | None = None,
) -> str:
    """Deterministic ID so re-ingesting the same thing upserts instead of duplicating.

    Keyed on the time span rather than a positional index: chunk 12 of one
    chunking run is a different moment than chunk 12 of another, so a
    positional id would silently rebind a vector to the wrong timeframe.

    `index` distinguishes several subjects sharing one chunk. `box_id` would
    read better but is not usable as a key: it is a within-chunk tracker
    referent, it repeats, and it is 0 for anyone the tracker missed.
    """
    key = f"{video_id}|{extractor_id}|{start:.3f}|{end:.3f}|{chunk_config}"
    if index is not None:
        key += f"|s{index}"
    digest = hashlib.sha1(key.encode()).hexdigest()
    return str(uuid.UUID(digest[:32]))


def config_key(
    preset: str | None,
    min_duration: float,
    max_duration: float,
    weights: dict[str, float] | None = None,
    interval: float | None = None,
) -> str:
    """Identifies which chunking scheme produced a vector, so two chunkings of
    the same video can coexist and be filtered apart.

    Custom weights are hashed into the key. Without that, two different
    weightings sharing a min/max would collide, and since point ids are derived
    from this key the second ingest would silently overwrite the first.
    """
    if interval is not None:
        return f"interval:{interval:g}"

    label = preset or "custom"
    if weights:
        digest = hashlib.sha1(
            "|".join(f"{k}={weights[k]:.4f}" for k in sorted(weights)).encode()
        ).hexdigest()[:8]
        label = f"{label}-{digest}"
    return f"{label}:{min_duration:g}-{max_duration:g}"


class ChunkStore:
    def __init__(self, path: str | None = None, model_name: str = DEFAULT_MODEL):
        ensure_dirs()
        self.client = QdrantClient(path=str(path or VECTOR_DIR))
        self.embedder = get_embedder(model_name)
        self._ensure_collection()

    def _ensure_collection(self):
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION not in existing:
            # One point per chunk carrying several named vectors, so a chunk
            # stays a single entity: limit=10 means 10 chunks, the payload is
            # stored once, and filters apply once.
            self.client.create_collection(
                COLLECTION,
                vectors_config={
                    field: VectorParams(size=self.embedder.dim, distance=Distance.COSINE)
                    for field in VECTOR_FIELDS
                },
            )
        if SUBJECTS not in existing:
            self.client.create_collection(
                SUBJECTS,
                vectors_config={
                    field: VectorParams(size=self.embedder.dim, distance=Distance.COSINE)
                    for field in SUBJECT_VECTOR_FIELDS
                },
            )
        # Embedded Qdrant ignores payload indexes and warns once per field.
        # Filtering still works correctly, it just scans rather than using an
        # index - fine at this scale. The calls are kept so the same code
        # gains real indexes if this ever points at a Qdrant server.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Payload indexes have no effect.*")
            for collection, fields in (
                (COLLECTION, INDEXED_FIELDS), (SUBJECTS, SUBJECT_INDEXED_FIELDS),
            ):
                for field, schema in fields.items():
                    try:
                        self.client.create_payload_index(
                            collection, field_name=field, field_schema=schema
                        )
                    except Exception:
                        pass  # already indexed

    def _embed_field_maps(
        self, field_maps: list[dict[str, str]]
    ) -> list[dict[str, list[float]]]:
        """Embed every field of every item in one batched GPU pass.

        The texts arrive as one {named_vector: text} map per point; they are
        flattened into a single batch and scattered back to their slots, so a
        video's whole index costs one pass rather than one per field per point.
        """
        flat = [
            (i, name, text)
            for i, fields in enumerate(field_maps)
            for name, text in fields.items()
        ]
        embedded = self.embedder.embed_documents([text for _, _, text in flat])
        vectors: list[dict[str, list[float]]] = [{} for _ in field_maps]
        for (i, name, _), vector in zip(flat, embedded):
            vectors[i][name] = vector.tolist()
        return vectors

    def add_chunks(
        self,
        video_id: str,
        video_path: str,
        chunks: list[dict],
        extractor_id: str,
        chunk_config: str,
    ) -> int:
        """Index one analyzer's output for every chunk of a video.

        Each chunk dict needs `start`, `end`, `output` (the analyzer's result)
        and `fields` ({named_vector: text}, produced by that analyzer's own
        render_fields). The store stays agnostic about analyzers so adding one
        never requires changing it. Payload carries everything needed to cite
        the moment, so retrieval never joins back to a JSON file.
        """
        points, field_maps = [], []
        for chunk in chunks:
            fields = chunk.get("fields") or {}
            if not fields.get("combined"):
                continue  # nothing to embed, e.g. a silent chunk's transcript
            points.append(chunk)
            field_maps.append(fields)

        if not points:
            return 0

        vectors_per_chunk = self._embed_field_maps(field_maps)

        structs = []
        for chunk, fields, vectors in zip(points, field_maps, vectors_per_chunk):
            output = chunk["output"] if isinstance(chunk["output"], dict) else {}
            text = fields["combined"]
            payload = {
                "video_id": video_id,
                "video_path": video_path,
                "extractor_id": extractor_id,
                "chunk_config": chunk_config,
                "chunk_id": chunk.get("id"),
                "start": float(chunk["start"]),
                "end": float(chunk["end"]),
                "text": text,
                "description": output.get("description", text),
                "setting": output.get("setting", ""),
                "people": output.get("people", []),
                "objects": output.get("objects", []),
                "actions": output.get("actions", []),
                "tags": output.get("tags", []),
                # Speaker-attributed analyzers: `speakers` is filterable
                # ("only chunks where SPEAKER_01 talks"), `turns` carries the
                # who-said-what breakdown so a result can be shown as a
                # dialogue without loading the record.
                "speakers": output.get("speakers", []),
                "turns": output.get("turns", []),
                # OCR: the read strings with what each one is.
                "texts": output.get("texts", []),
                # Object detection: each object with its appearance and use.
                "detections": output.get("detections", []),
                # People: a count for range filters that semantic search cannot
                # answer. The per-person records themselves are already in
                # `people` above - this used to also write them to `persons`,
                # a byte-identical copy under a second name.
                "people_count": output.get("people_count"),
            }
            structs.append(
                PointStruct(
                    id=point_id(video_id, extractor_id, chunk["start"], chunk["end"], chunk_config),
                    vector=vectors,
                    payload=payload,
                )
            )

        self.client.upsert(COLLECTION, points=structs)
        return len(structs)

    def add_subjects(
        self,
        video_id: str,
        video_path: str,
        subjects: list[dict],
        extractor_id: str,
        chunk_config: str,
    ) -> int:
        """Index one point per subject, as produced by an analyzer's render_subjects.

        Each subject dict needs `chunk_id`, `start`, `end`, `index`, `vectors`
        ({named_vector: text}) and `payload`. The store stays agnostic about
        which analyzer produced them, exactly as add_chunks does - an analyzer
        gains subject-level search by growing one method, changing nothing here.
        """
        usable = [s for s in subjects if s.get("vectors", {}).get(SUBJECT_VECTOR_FIELDS[0])]
        if not usable:
            return 0

        vectors_per_subject = self._embed_field_maps([s["vectors"] for s in usable])

        structs = []
        for subject, vectors in zip(usable, vectors_per_subject):
            structs.append(
                PointStruct(
                    id=point_id(
                        video_id, extractor_id, subject["start"], subject["end"],
                        chunk_config, subject["index"],
                    ),
                    vector=vectors,
                    payload={
                        "video_id": video_id,
                        "video_path": video_path,
                        "extractor_id": extractor_id,
                        "chunk_config": chunk_config,
                        "chunk_id": subject["chunk_id"],
                        "start": float(subject["start"]),
                        "end": float(subject["end"]),
                        "subject_index": subject["index"],
                        **subject["payload"],
                    },
                )
            )

        self.client.upsert(SUBJECTS, points=structs)
        return len(structs)

    def _query(
        self,
        collection: str,
        fields: tuple[str, ...],
        spec: dict,
        query: str | list[float],
        limit: int,
        score_threshold: float | None,
        field: str,
        filters: dict,
    ) -> list[dict]:
        """Vector search against one collection, narrowed by its own filters.

        Both collections search the same way and differ only in which named
        vectors and which filter names are legal, so those arrive as arguments
        rather than as a second copy of this method.
        """
        if field not in fields:
            raise ValueError(f"Unknown vector field {field!r}; expected one of {list(fields)}")
        must, must_not = _conditions(filters, spec)
        vector = query if isinstance(query, list) else self.embedder.embed_query(query).tolist()

        response = self.client.query_points(
            collection,
            query=vector,
            using=field,
            query_filter=Filter(must=must, must_not=must_not) if (must or must_not) else None,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [{"score": p.score, "point_id": p.id, **p.payload} for p in response.points]

    def search_subjects(
        self,
        query: str | list[float],
        limit: int = 10,
        score_threshold: float | None = None,
        field: str = "clothing",
        **filters,
    ) -> list[dict]:
        """Nearest people, one hit per person rather than per chunk.

        `query` is text, or an already-embedded vector - which is what makes
        "who else looks like this person" a single call rather than a re-embed
        of a description the store already holds.
        """
        return self._query(
            SUBJECTS, SUBJECT_VECTOR_FIELDS, SUBJECT_FILTER_SPEC,
            query, limit, score_threshold, field, filters,
        )

    def get_subject(self, point_id: str, with_vectors: bool = True) -> dict | None:
        """One subject point by id, with its vectors, for search-by-example."""
        found = self.client.retrieve(
            SUBJECTS, ids=[point_id], with_payload=True, with_vectors=with_vectors
        )
        if not found:
            return None
        point = found[0]
        return {
            "point_id": point.id,
            **point.payload,
            **({"vectors": point.vector} if with_vectors else {}),
        }

    def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float | None = None,
        field: str = "combined",
        **filters,
    ) -> list[dict]:
        """Vector search narrowed by any combination of payload filters.

        Cosine similarity always ranks *something*, so a query for content a
        video does not contain still returns its nearest neighbours rather
        than nothing. `score_threshold` is what makes "no match" expressible.
        On this data present content scored >=0.665 and absent content <=0.524,
        so ~0.55-0.60 separates them; retune per embedding model.

        `field` picks the named vector space: "combined" (default) searches the
        whole flattened record, while "people"/"actions"/"objects"/"description"
        match against just that part. Chunks lacking the chosen field are
        naturally excluded rather than matched on empty text.
        """
        return self._query(
            COLLECTION, VECTOR_FIELDS, FILTER_SPEC,
            query, limit, score_threshold, field, filters,
        )

        return [{"score": p.score, **p.payload} for p in response.points]

    def delete_video(
        self,
        video_id: str,
        chunk_config: str | None = None,
        extractor_id: str | None = None,
    ) -> None:
        """Drop a video's vectors.

        Narrow with `chunk_config` and/or `extractor_id`. Re-ingest should
        scope to the analyzer being re-run: deleting by video alone wipes
        every *other* analyzer's vectors too, so adding one analyzer to an
        existing video would silently destroy the rest.
        """
        must = [FieldCondition(key="video_id", match=MatchValue(value=video_id))]
        if chunk_config:
            must.append(FieldCondition(key="chunk_config", match=MatchValue(value=chunk_config)))
        if extractor_id:
            must.append(FieldCondition(key="extractor_id", match=MatchValue(value=extractor_id)))
        # Both collections: a subject point describes people inside the chunks
        # being dropped, so leaving it would strand it against a chunk that no
        # longer exists.
        for collection in (COLLECTION, SUBJECTS):
            self.client.delete(collection, points_selector=Filter(must=must))

    def count(self, collection: str = COLLECTION) -> int:
        return self.client.count(collection).count

    def close(self) -> None:
        """Release the local storage lock. Qdrant's embedded client otherwise
        closes during interpreter shutdown and logs a noisy (harmless)
        ImportError."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
