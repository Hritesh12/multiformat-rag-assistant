# Build Log — real bugs found and fixed during development

This file exists so the debugging story is documented as it happened,
not reconstructed afterward. Each entry is something that was actually
broken, how it was caught, and the fix — useful for explaining the
project honestly in an interview.

## 1. module_level code chunk swallowed the entire file

**Where:** `src/ingest/code_chunker.py`, `_extract_module_level`

**Bug:** Initial implementation spanned from the first non-def/class
top-level node's line number to the last one's line number, and took
every line in between. But function and class bodies defined *between*
those two nodes are also "in between" line-number-wise — so the
"module level" chunk ended up containing the entire file, duplicating
every function and class a second time under the wrong label.

**Caught by:** Direct test against a sample file with functions, a
class with methods, and a __main__ block — printed all chunks and
saw the module_level chunk was 37 lines when it should have been ~7.

**Fix:** Collect each qualifying top-level node's own line range
individually and concatenate those, instead of spanning from first
to last. Skips over any def/class bodies in between correctly.

## 2. PDF chunker produced one 4945-character chunk on a real resume

**Where:** `src/ingest/pdf_chunker.py`, `_split_paragraphs`

**Bug:** The chunker had two fallback layers: blank-line paragraphs,
then sentence-boundary splitting if no blank lines existed. Tested
against a real uploaded resume PDF (not a synthetic test file) and
got 1 chunk for the entire page, badly exceeding the 1500-char max.

**Root cause:** The resume has zero `\n\n` blank lines (dense, single-
column layout) AND long stretches with no `. ` / `! ` / `? ` sentence-
terminal punctuation (bullet fragments, pipe-separated headers,
skill lists). Both fallback layers found nothing to split on, so the
whole page collapsed into a single "sentence."

**Caught by:** Running the chunker against real uploaded resume PDFs
and checking `max(len(c.content) for c in chunks)` against the
configured bound — this is why testing against real, messy documents
matters more than testing against clean synthetic ones.

**Fix:** Added a third fallback layer — group by single newlines
(3 lines at a time) when neither blank-lines nor sentence punctuation
give usable structure. Re-verified against 3 different resume PDFs
after the fix; all stayed within bound.

## 3. CSV schema summary listed every unique value for ID-like columns

**Where:** `src/ingest/csv_handler.py`, `_build_schema_chunk`

**Bug:** A column was treated as "categorical" (and every value
listed) if it had <=20 unique values — which is true for a small
test CSV's `restaurant` name column (7 rows, 7 unique names), even
though a name/ID column isn't actually categorical and this gets
worse, not better, as a real dataset grows to thousands of rows.

**Fix:** Added a check — a column where every value is unique
(`nunique() == len(df)`) is labeled as an identifier column instead,
and its values aren't dumped into the summary.

## 4. ChromaDB EphemeralClient instances share state by collection name

**Where:** `tests/test_backend_logic.py`, discovered while testing
`chroma_bm25_backend.py`

**Bug:** Test helper `make_test_backend()` created a "new"
`chromadb.EphemeralClient()` for every test function, all using the
same hardcoded collection name. Running the full test file, the
reindex test reported `collection.count() == 3` when only 1 document
had ever been indexed within that test.

**Root cause, confirmed directly:** two separate `EphemeralClient()`
instances, given the same collection name, share underlying state
within a process — adding a document via one client's collection made
it immediately visible via `.count()` on the OTHER client's
same-named collection, with no `.add()` call made on that second
client at all. This isn't documented behavior I assumed; I tested it
directly with a 6-line isolated reproduction before accepting it as
the explanation, specifically to rule out the alternative hypothesis
that the bug was in my own indexing/upsert logic.

**Fix:** Generate a unique collection name (`uuid4` suffix) per test
backend instance, so tests are actually isolated from each other.
Documented on the production class too, since anyone extending this
into multi-session use needs to know collection names must be unique
per session.

## 5. keyword_search filtered out true matches on small corpora

**Where:** `src/retrieval/chroma_bm25_backend.py`, `keyword_search`

**Bug:** `keyword_search("updated")` against a 1-document corpus
containing "updated" returned zero results — even though the term is
present.

**Root cause:** the filter logic assumed `score <= 0` meant "no
query-term overlap" and discarded those results. That assumption is
false: BM25's IDF term can go negative when a term appears in a large
fraction of the corpus, which is guaranteed for EVERY term in a
single-document corpus (verified directly: scored -0.27 for an exact
match). The negative score reflects that BM25's ranking signal is
weak/inverted in this regime, not that there was no match.

**Fix:** Filter on ACTUAL TOKEN OVERLAP between query and document,
not on score sign. Score is still used for ranking among confirmed
matches; overlap decides whether something is a match at all. This is
a good example of a subtle-but-real algorithmic bug that only showed
up by testing an edge case (tiny corpus) directly, rather than only
testing with a "normal-sized" 3-5 document corpus where BM25's scores
behave more intuitively.

## 6. Router misclassified code questions containing numeric words

**Where:** `src/retrieval/router.py`, `route`

**Bug:** "What is the maximum recursion depth in this function?" routed
to COMPUTE instead of CODE, because compute patterns were checked
unconditionally before code patterns, and "maximum" matched first —
even though "this function" makes clear the question is about code,
not a tabular aggregate.

**Caught by:** Deliberately testing adversarial queries that mix a
compute-sounding word with a clear code subject, rather than only
testing clean unambiguous examples (which all passed and would have
hidden this).

**Fix:** Added a `_CODE_SUBJECT_PATTERNS` check (phrases like "this
function", "in this codebase") that runs BEFORE compute patterns.
When the query has a clear code subject, code wins regardless of
incidental numeric words. This doesn't make the router perfect — it's
still rule-based and will miss phrasings outside these patterns — but
it fixes the specific class of failure found during testing.

## 7. matched_pattern displayed raw regex syntax instead of matched text

**Where:** `src/retrieval/router.py`, `_first_match`

**Bug:** `matched_pattern` was showing values like `"max(imum)?"` and
`"how many"` with escape artifacts, instead of clean text — because
`pattern.strip(r"\b")` and `.replace(r"\d+", "N")` were called on the
REGEX PATTERN STRING, and `str.strip()`/`str.replace()` only operate
on literal characters, not regex syntax. They silently did nothing
useful and the raw pattern leaked through to what's shown in the UI.

**Fix:** Use `re.search(...).group(0)` to get the actual substring
matched IN THE QUERY TEXT, not the pattern used to find it. Simple
fix, but the kind of bug that's easy to miss because it doesn't crash
— it just silently shows ugly/wrong text, which is why explicitly
printing and reading the output during testing mattered here.

## 8. Compute executor picked the wrong column on ambiguous matches

**Where:** `src/retrieval/compute_executor.py`, `_match_column` /
`_run_operation`

**Bug (found in two related stages):**

Stage 1 — "What is the average rating by zone?" crashed with "'zone'
is not numeric, so mean isn't a valid operation on it." The query
clearly wants mean(rating) grouped by zone, but `_match_column`
returned `zone` (the groupby key) as the PRIMARY column, because it
iterates `df.columns` in file order and `zone` appears before `rating`
in this CSV, and both literally appear in the query text.

Stage 2 — after fixing stage 1 (resolve the "by X" groupby column
first, exclude it from primary-column matching), a DIFFERENT query
regressed: "How many restaurants have online ordering?" started
returning a useless per-restaurant count of 1 each, instead of a
Yes/No breakdown. Cause: "restaurants" (plural, in the query) contains
"restaurant" (singular, an identifier-like column with one unique
value per row) as a substring, and `restaurant` sorts before
`online_ordering` in column order — so it matched first, even though
the question is clearly about `online_ordering`.

**Caught by:** Running the full query set after EACH fix, not just
the one query being fixed — this is what caught the stage-2
regression. Fixing stage 1 in isolation and moving on would have
shipped a new bug.

**Fix:** Two changes, both needed together: (1) resolve any "by X" /
"per X" groupby column first and exclude it from primary-column
matching; (2) when multiple columns match the query text, prefer
non-identifier-like columns (nunique < row count) over identifier-like
ones (nunique == row count) — a query casually mentioning the
dataset's subject in passing shouldn't out-rank a real categorical
match. Documented as a heuristic, not a guarantee, since a dataset
where the identifier column genuinely IS the right target would still
mismatch — but it resolves the concrete cases found in testing.

## 9. Identifier-column heuristic wrongly deprioritized numeric columns

**Where:** `src/retrieval/compute_executor.py`, `_match_column`

**Bug:** Full end-to-end pipeline test (real ingestion -> real
routing -> real compute) failed on "What is the average cost across
all restaurants?" — a query nearly identical in structure to ones
already passing in isolated ComputeExecutor tests. Error: 'restaurant'
is not numeric, so mean isn't a valid operation on it.

**Root cause:** Bug #8's fix deprioritized "identifier-like" columns
using `nunique() == len(df)` (every value is unique). That heuristic
is correct for a genuine identifier column (a name), but WRONG for a
numeric column: in the small 7-row test dataset, `cost` also happens
to have 7 unique values (450, 320, 550, 600, 700, 200, 900 — no
coincidental duplicates), so it was ALSO flagged as identifier-like
and deprioritized — putting it on equal footing with `restaurant`
instead of being correctly preferred. Since `restaurant` came first
in column order among the (wrongly) equally-deprioritized set, it won.

This is exactly why integration testing matters beyond unit testing:
the isolated ComputeExecutor tests for bug #8 happened to use queries
where this distinction didn't matter, so they kept passing while a
real end-to-end query with a different (but structurally similar)
phrasing exposed the flaw.

**Fix:** The deprioritization now only applies to NON-numeric columns.
`pd.api.types.is_numeric_dtype(df[col])` is checked first — numeric
columns are never deprioritized by the identifier heuristic, since
cardinality alone can't distinguish "identifier" from "continuous
measurement" (a cost, rating, or timestamp column is EXPECTED to have
high or total cardinality without being an identifier). Re-ran both
the full isolated compute test set AND the end-to-end suite together
after this fix — both pass simultaneously, confirming no regression.
