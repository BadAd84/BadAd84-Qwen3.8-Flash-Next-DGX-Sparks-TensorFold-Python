"""Flash Next's concurrent rounds on one GPU: every stream keeps exactly its own accepted prefix."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from tensorfold.cuda.logprobs import capture

from tensorfold.cuda.capacity import available_bytes
from tensorfold.cuda.markers import MIN_GAP
from tensorfold.cuda.memory_gate import MemoryGate, NoRoom, torch_live
from tensorfold.cuda.sampling import sample_streams
from tensorfold.cuda.streams import Stream, accept
from tensorfold.engine.exact_sampling import MARGIN, choose_rows
from tensorfold.engine.grammar import GrammarError

from .decode import PREFILL_ROWS, WARM_TAIL, Engine, absorb, draft, entry_end, prefill_begin
from . import attn_multi, gdn_multi, image_rows, prefixes
from .copies import Copies, fit
from .forward import Cut, commit, compute, compute_mixed, converges, cut_snapshot, read_ahead, stage
from .mtp import mtp_compute, mtp_stage
from .prompt_plan import pass_limit
from .state import ENDS, Buffers, State
from ..cuda import CONFIDENCE, COPY_ROWS, DEPTH

FIRST, STEP = 256, 8192          # rows an idle slot keeps; rows a stream's caches grow by at a time
GIB = 1024**3
SHARE = 0.0                      # --decode-share: a round alone takes this share of its pass's time (0: whole passes)
PASS_MIN = 128                   # the fewest prompt rows a round's pass takes
COPY_MATCH = 8                   # copy drafts: the context's last this many tokens seen before, and as many after them
FILL_GUARD = 8                   # a prompt passed over this many passes takes the next one (no starvation), as on Macs
HINTS = os.environ.get("TENSORFOLD_PLE_HINTS", "1") != "0"   # the next round's n-gram rows read while the GPU drafts
# Where an MTP chain stops (drafts change speed only, never the reply's bits), TENSORFOLD_CHAIN_RULE:
#   "conf" (default): a draft under --mtp-confidence ends it, as v0.6.1;
#   "cum": the chance every draft so far lands (their calibrated probabilities' product) falls under TENSORFOLD_CHAIN_T;
#   "cost": a draft stays while its chance of landing buys more than a verify row, the chain goes on while the next
#     one's would also pay for an MTP step; "adapt" / "acum": "cost" / "cum" with each stream's own recent landing
#     rate scaling its drafts' calibrated chances.
# Measured on real prose and code at 1/2/4 streams (scripts/qwen38/README.md): "cum" at 0.30 with --mtp-drafts 8.
CHAIN_RULE = os.environ.get("TENSORFOLD_CHAIN_RULE", "conf")
CHAIN_T = float(os.environ.get("TENSORFOLD_CHAIN_T", "0.30"))
ROW_MS = (3.1, 2.2)              # a verify row's cost at one stream and at four or more (measured on GB10)
MTP_MS = 1.4                     # an MTP step's cost (the draft head's 128 MB read dominates it)
# a draft's chance of landing given every earlier draft landed, by its draft-head probability: (probability, landing
# rate) points, linear between them and flat past the ends. Measured greedy on real prose and code, every chain run to
# depth 8 (4,935 tested drafts in bins of 0.05 holding 40 or more), fit monotone (pooled adjacent violators).
CAL = ((0.1304, 0.087), (0.1766, 0.1644), (0.2266, 0.1667), (0.2757, 0.2014), (0.3496, 0.2612), (0.4255, 0.277),
       (0.5026, 0.4088), (0.5757, 0.4759), (0.6267, 0.5128), (0.6982, 0.5813), (0.7753, 0.6597), (0.8522, 0.7105),
       (0.9268, 0.7479), (0.9932, 0.9507))


def _hinted(w) -> bool:
    """Whether any layer's n-gram table reads from disk at each lookup and takes page hints (--ple-on-ssd)."""

    return HINTS and getattr(w, "x3", None) is None and any(
        getattr(getattr(layer.ple, "table", None), "will_need", None) is not None
        for layer in w.layers if layer.ple is not None)


def _will_need(w, rows: list) -> None:
    """Start the page reads of the n-gram rows a coming ``stage`` looks up: (history, tokens, last) each, the rows of
    the ``last`` final tokens after ``history`` (hints: ``stage`` reads the same bytes whether or not they ran)."""

    for layer in w.layers:
        ple = layer.ple
        hint = getattr(getattr(ple, "table", None), "will_need", None) if ple is not None else None
        if hint is None:
            continue
        for history, tokens, last in rows:
            hint(ple.ngram.ids(history, np.asarray(tokens, dtype=np.int64))[-last:])


def _slot(w, st: State, buf: Buffers, mbuf: Buffers, pbuf: Buffers, capacity: int, prefill_rows: int) -> Engine:
    """A one-sequence engine over a slot's state and the shared buffers (eager: no CUDA graphs)."""

    e = object.__new__(Engine)
    e.w, e.capacity, e.rows, e.prefill_rows = w, capacity, buf.rows, prefill_rows
    e.buf, e.mbuf, e.pbuf, e.st, e.graphs = buf, mbuf, pbuf, st, None
    e.stops = ()
    return e


class MultiDecoder:
    """Rounds over the live streams; ``slots`` streams at most, each with ``capacity`` tokens of context."""

    def __init__(self, w, *, slots: int, capacity: int, depth: int = DEPTH, confidence: float = CONFIDENCE,
                 stop_eos: bool = True, keep: int = 8, kv_dtype: str = "bf16", prefill_rows: int = PREFILL_ROWS,
                 share: float = SHARE, points=None, vision=None, workspace_bytes: int = 0,
                 copy_rows: int = COPY_ROWS) -> None:
        if w.comm is not None:
            raise ValueError("concurrent Flash Next runs on one GPU for now")
        self.w, self.depth, self.confidence, self.capacity = w, depth, confidence, capacity
        self.points = points                         # a prompt's message starts to keep states at, or None
        self.vision = vision
        self.copy_rows = copy_rows if depth > 0 and w.comm is None else 0   # a copy's widest window (0: none)
        self.eos = tuple(w.cfg.eos) if stop_eos else ()
        rows = max(slots * (depth + 1), self.copy_rows)  # a chain's rows a stream; copies share what chains leave
        # a round's window and a prompt pass share each layer's expert launch: the pass's buffers hold both
        self.converged, self.prefill_rows = converges(w), prefill_rows
        # rounds beside a filling prompt size its pass so decoding keeps ``share`` of the pass's time (0: whole passes)
        self.share, self.round_s, self.row_s = share, None, None
        self.buf = Buffers(w, rows, capacity, moe_prefill=True)
        self.mbuf = Buffers(w, rows, capacity) if w.mtp is not None else None
        self.pbuf = Buffers(w, prefill_rows + (rows if self.converged else 0), capacity, prefill=True)
        self.gdn = gdn_multi.Scratch(w, rows)            # every stream's DeltaNet rows, one launch a step
        self.held: dict[int, list[int]] = {}             # stream id -> last round's kept rows, folded in next round
        # slots start small and grow with their stream's context, up to the window, while the gate has room
        # every slot's replay rows hold a copy: any slot may host one stream's wide window
        self.free = [State(w, min(capacity, FIRST), max(depth + 1, self.copy_rows), kv_dtype, limit=capacity)
                     for _ in range(slots)]
        self.slot_bytes = sum(t.numel() * t.element_size() for t in _tensors(self.free[0]))
        self.window_bytes = self.free[0].cache_bytes(capacity)          # one stream's caches at the full window
        free = torch_live(torch, available_bytes) if torch.cuda.is_available() else None
        # the mapped n-gram tables are not held back (they barely fit on a Spark); lookups page from disk instead
        live = free
        self.memory_gate = MemoryGate(live() if live is not None else 1 << 62,
                                      reserve=max(2 * GIB, workspace_bytes), live=live)
        self.streams: dict[int, Stream] = {}
        self.filling: list[Stream] = []                  # admitted, prompts still prefilling (oldest first)
        self.fills: dict[int, list] = {}                 # stream id -> [its engine, drafts?, next row, kept state]
        self.passed: dict[int, int] = {}                 # stream id -> prompt passes since its last rows
        self.arrived = lambda: False                     # a request waits to be admitted (the scheduler sets this)
        self.next_id = 0
        self.draft_host = w.draft_ids.cpu().numpy() if w.draft_ids is not None else None
        self.kept: list[tuple[list[int], State, dict, torch.Tensor | None]] = []   # (ids, slot, snapshot, tail)
        self.keep = keep
        # the next round's n-gram rows (its first token, then each draft) start reading on this thread as they are
        # known, so the round's stage finds their pages cached; None: tables in memory, or hints off
        self.hints = ThreadPoolExecutor(1, thread_name_prefix="ple-hint") if _hinted(w) else None
        self.chain_rule, self.chain_t = CHAIN_RULE, CHAIN_T
        self.tok_ms = None                               # round ms a token kept, all streams (the cost rule's value)
        self.verdict = None                              # verdict(stream, draft probabilities, drafts landed): a probe

    def _busy(self) -> set[int]:
        return {id(s.st) for s in [*self.streams.values(), *self.filling]}

    def _drop_kept(self, st: State) -> None:
        self.kept = [k for k in self.kept if k[1] is not st]

    def _grow(self, st: State, rows: int, *, alone: bool = False, protect: State | None = None) -> bool:
        """Grow caches to hold ``rows`` while the gate has room, kept ends first; ``alone`` grows anyway."""

        if rows <= st.capacity or st.capacity >= st.limit:       # admission's count keeps a stream within its window
            return True
        size = min(st.limit, -(-rows // STEP) * STEP)
        before = st.cache_bytes()
        grow = st.cache_bytes(size) - before
        while not self.memory_gate.fits(grow + st.layer_bytes(size)):     # a layer's old buffers stay until its copy
            if not self._evict_kept(st, protect=protect):
                if alone:
                    break
                return False
        try:
            added = st.resize(size)
        except Exception:
            self.memory_gate.take(st.cache_bytes() - before)
            raise
        self.memory_gate.take(added)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()             # the old buffers back to the system: MemAvailable stays true
        return True

    def _shrink(self, st: State, *, force: bool = False) -> None:
        """An idle slot back to its first rows: its caches' memory returns to the gate."""

        st.reset(self.w)
        if force or st.capacity > FIRST:
            self.memory_gate.give(-st.resize(min(FIRST, st.limit)))

    def _evict_kept(self, keep: State, *, protect: State | None = None) -> bool:
        """Free the oldest idle kept prompt end (never ``keep``); False when none is left."""

        busy = self._busy()
        for ids, st, _, _ in self.kept:
            if st is not keep and st is not protect and id(st) not in busy:
                self._drop_kept(st)
                self._shrink(st)
                if all(f is not st for f in self.free):
                    self.free.append(st)
                return True
        return False

    def _make_room(self) -> list[Stream]:
        """Before a round: grow each live window oldest-first; a stream that can't grow makes the newest end."""

        live = sorted((s for s in self.streams.values() if not s.done), key=lambda s: s.sid)
        blocked = False
        for s in live:
            rows = max(s.st.pos, s.st.mtp_len) + len(s.drafts) + self.depth + 2
            s.waiting = rows > s.st.capacity if blocked else not self._grow(s.st, rows, alone=len(live) == 1)
            blocked = blocked or s.waiting
        if live and live[0].waiting and len(live) > 1:        # even the oldest can't grow: the newest ends
            newest = live[-1]
            newest.error = RuntimeError(
                f"This server ran out of memory with {len(live)} streams decoding, so the newest (this request, after "
                f"{len(newest.out)} tokens) was stopped for the older ones to finish. Retry it, shorten the prompt or "
                "max_tokens, or start the server with a smaller --parallel.")
            newest.done, newest.waiting = True, False
            self.memory_gate.ends += 1
            self.streams.pop(newest.sid, None)
            self.held.pop(newest.sid, None)
            self._drop_kept(newest.st)
            self._shrink(newest.st)
            self.free.append(newest.st)
            return [newest, *self._make_room()]
        self.memory_gate.waits += any(s.waiting for s in live)
        return []

    def _slot_for(self, prompt: list[int], reuse: bool):
        """Reuse the longest kept point, copying a fork into a free slot when the memory gate permits it."""

        return prefixes.slot_for(self, prompt, reuse)

    def _remember(self, ids: list[int], st: State, snap: dict, tail) -> None:
        prefixes.remember(self, ids, st, snap, tail)

    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    @torch.no_grad()
    def warm(self) -> None:
        """A synthetic greedy request through prefill (a full chunk, then a partial one cut at the kept point), its drafts and one round, then forgotten, so no request compiles or loads a kernel."""

        s = Stream([0] * min(self.prefill_rows + WARM_TAIL + 1, self.capacity - self.depth - 2), 2)
        self.admit(s)
        if not s.done:
            self.round()                                 # the whole prompt (nothing else decodes), then a round
        self.streams.pop(s.sid, None)
        self._drop_kept(s.st)
        self._shrink(s.st)
        if all(f is not s.st for f in self.free):
            self.free.append(s.st)

    @torch.no_grad()
    def admit(self, s: Stream) -> None:
        """Queue a request in a free slot (a kept prompt end it extends, if any); rounds prefill its prompt."""

        room = self.capacity - len(s.prompt) - self.depth - 1
        if room < 1:
            raise ValueError(f"a prompt of {len(s.prompt)} tokens leaves no room in the {self.capacity}-token context")
        s.count = max(1, min(s.count, room))
        if any(x.waiting for x in self.streams.values()):
            raise NoRoom("streams already wait for memory; a new request waits until one finishes")
        t0 = time.perf_counter()
        st, resume, s.cached = self._slot_for(list(s.prompt), s.draft and s.vision is None)
        if not self._grow(st, len(s.prompt) + self.depth + 2, alone=not self.streams and not self.filling):
            if resume is None:
                self.free.append(st)
            else:                                        # the kept prompt end stays kept
                self._remember(list(s.prompt[:s.cached]), st, resume["state"], resume["tail"])
            raise NoRoom(f"a {len(s.prompt)}-token prompt waits for memory until a live stream finishes")
        e = _slot(self.w, st, self.buf, self.mbuf, self.pbuf, self.capacity, self.prefill_rows)
        mtp = s.draft and self.depth > 0 and self.mbuf is not None
        try:
            begin = prefill_begin(e, s.prompt, mtp=mtp, resume=resume)
            image = s.vision is not None
            image_rows.begin(e, s, self.vision)
            if image:
                torch.cuda.empty_cache()                 # the tower's scratch back before the prompt's passes
        except Exception:
            self._drop_kept(st)
            self.free.append(st)
            raise
        e.stops = sorted({p for p in self.points(s.prompt) if begin + MIN_GAP <= p < entry_end(s.prompt)}) \
            if s.draft and st.image_positions is None and self.points is not None else []
        s.sid, s.st = self.next_id, st
        self.next_id += 1
        s.prefill_s = time.perf_counter() - t0
        same = resume is not None and self._keep_at(s) == begin         # the same prompt again: its own point
        self.fills[s.sid] = [e, mtp, begin, (resume["state"], resume["tail"]) if same else None]
        self.filling.append(s)

    def _fill(self) -> list[Stream]:
        """Fill ordered prompt pieces until a stream decodes or a waiting request needs admission."""

        ended: list[Stream] = []
        while self.filling:
            ended += self._pass()
            if any(not x.done and not x.waiting for x in self.streams.values()) or self.arrived():
                break
        return ended

    def _pass_rows(self) -> int:
        """A round's prompt rows: its decode (a round alone) takes ``share`` of the pass's time, by the last rounds."""

        live = any(not s.done for s in self.streams.values())
        return pass_limit(self.prefill_rows, live, self.share, self.round_s, self.row_s, PASS_MIN)

    def _timed(self, seconds: float, rows: int) -> None:
        """A round's wall time: a round alone updates its estimate, a round with a pass the seconds a row adds."""

        if rows:
            extra = max(0.0, seconds - (self.round_s or 0.0)) / rows
            self.row_s = extra if self.row_s is None else 0.7 * self.row_s + 0.3 * extra
        else:
            self.round_s = seconds if self.round_s is None else 0.7 * self.round_s + 0.3 * seconds

    def _order(self) -> list[Stream]:
        """Order overdue prompts first, then foreground, fewest rows left and stable arrival order."""

        def key(s: Stream):
            passed = self.passed.get(s.sid, 0)
            due = passed >= FILL_GUARD
            return (not due, -passed if due else 0, s.background, len(s.prompt) - self.fills[s.sid][2])

        return sorted(self.filling, key=key)                         # stable: ties keep arrival order

    def _pieces(self, rows: int | None = None) -> list[tuple[Stream, int, int]]:
        """The next pass: rows from the filling prompts in ``_order``, up to ``rows`` and ENDS ending prompts."""

        pieces, room = [], self._pass_rows() if rows is None else rows
        for s in self._order():
            e, mtp, start, _ = self.fills[s.sid]
            n = min(next((p for p in e.stops if p > start), len(s.prompt)) - start, room)
            ends = sum(1 for x, a, k in pieces if a + k == len(x.prompt))
            if n == 0 or (start + n == len(s.prompt) and ends == ENDS):
                break
            pieces.append((s, start, n))
            room -= n
        return pieces

    def _pass(self) -> list[Stream]:
        """One prompt pass alone; prompts that end sample their first token, draft and join the rounds."""

        pieces = self._pieces()
        self._note_passed(pieces)
        t0 = time.perf_counter()
        try:
            self._read_ahead(pieces)
            segs = stage(self.w, self.pbuf, [(s.st, s.prompt[a:a + n]) for s, a, n in pieces])
            ends, cuts = self._end_rows(pieces, segs), self._cuts(pieces, segs)
            logits = compute(self.w, segs, self.pbuf, logits=bool(ends), ends=ends, cuts=cuts)
            heads = logits[:len(ends)].clone() if ends else None
            lasts = self._absorb(pieces, segs, cuts)
        except Exception as exc:                         # noqa: BLE001  (these requests fail, the others go on)
            return self._failed(pieces, exc)
        return self._joined(pieces, heads, lasts, (time.perf_counter() - t0) / len(pieces))

    def _read_ahead(self, pieces) -> None:
        """Overlap the next piece's n-gram reads with this pass."""

        ngram = self.w.cfg.ngram_size - 1
        for s, a, n in pieces:
            if a + n >= len(s.prompt) or s.st.ple_history is None:
                continue
            hist = np.concatenate([np.asarray(s.st.ple_history, dtype=np.int64),
                                   np.asarray(s.prompt[a:a + n], dtype=np.int64)])[-ngram:]
            read_ahead(self.w, hist, s.prompt[a + n:a + n + self.prefill_rows])

    def _note_passed(self, pieces) -> None:
        """Count a pass against every filling prompt it left out; one it took starts over."""

        took = {s.sid for s, _, _ in pieces}
        for s in self.filling:
            self.passed[s.sid] = 0 if s.sid in took else self.passed.get(s.sid, 0) + 1
        for sid in [k for k in self.passed if k not in {s.sid for s in self.filling}]:
            del self.passed[sid]

    @staticmethod
    def _end_rows(pieces, segs) -> list[int]:
        """The pass rows that end a prompt (each gets the head)."""

        return [a1 - 1 for (s, a, n), (_, _, a1) in zip(pieces, segs) if a + n == len(s.prompt)]

    @staticmethod
    def _keep_at(s: Stream) -> int | None:
        """Where a drafting stream's prompt state is kept: one token before its end, which a next turn extends."""

        return entry_end(s.prompt) if s.draft and s.st.image_positions is None else None

    def _point(self, s: Stream, start: int) -> int | None:
        """The next message-start or prompt-end snapshot this prompt piece can reach."""

        if not s.draft or s.st.image_positions is not None:
            return None
        return next((p for p in self.fills[s.sid][0].stops if p > start), self._keep_at(s))

    def _cuts(self, pieces, segs) -> list[Cut]:
        """The kept points strictly inside the pass's pieces, where their DeltaNet chains split."""

        return [Cut(k - a, at=a0) for (s, a, n), (_, a0, _) in zip(pieces, segs)
                if (k := self._point(s, a)) is not None and a < k < a + n]

    def _absorb(self, pieces, segs, cuts=()) -> list[torch.Tensor]:
        """After a pass's forward: each prompt's last row and kept point, the MTP head's absorb, the commits."""

        lasts = [self.pbuf.streams[a1 - 1:a1].clone() for _, _, a1 in segs]
        at, points = {cut.at: cut for cut in cuts}, []
        for (s, a, n), (st, a0, _) in zip(pieces, segs):      # before the MTP head writes the pass's streams
            k = self._point(s, a)
            if k is None or not a < k <= a + n:
                continue
            row, mtp = k - a, self.fills[s.sid][1]
            mtp_len = st.mtp_len + row - 1 if mtp else st.mtp_len       # every row but the point's last
            tail = self.pbuf.streams[a0 + row - 1:a0 + row].clone() if mtp else None
            cut = at.get(a0)
            snap = None if cut is None else cut_snapshot(self.w, st, self.pbuf, cut, mtp_len)
            points.append((s, mtp_len, tail, snap))
        absorb = [(s.st, s.prompt[a + 1:a + n + 1], self.pbuf.streams[a0:a0 + n])
                  for (s, a, n), (_, a0, _) in zip(pieces, segs) if self.fills[s.sid][1] and a + 1 < len(s.prompt)]
        if absorb:                   # the MTP head absorbs each prompt's rows (its cache in position order)
            absorb = [(st, nxt, streams[:len(nxt)]) for st, nxt, streams in absorb]
            mtp_compute(self.w, mtp_stage(self.w, self.pbuf, absorb), self.pbuf, logits=False)
            for st, nxt, _ in absorb:
                st.set_mtp_len(st.mtp_len + len(nxt))
        for (s, a, n), (st, a0, _) in zip(pieces, segs):
            commit(self.w, st, self.pbuf, n, n, at=a0)
        for s, mtp_len, tail, snap in points:            # a point that ends its piece: the state as committed
            self.fills[s.sid][3] = (snap if snap is not None else {**s.st.snapshot(), "mtp_len": mtp_len}, tail)
        return lasts

    def _failed(self, pieces, exc: Exception) -> list[Stream]:
        failed = [s for s, _, _ in pieces]
        for s in failed:
            s.error, s.done = exc, True
            self.filling.remove(s)
            self.fills.pop(s.sid)
            self._drop_kept(s.st)
        return failed                                    # finish() frees their slots

    def _joined(self, pieces, heads, lasts, spent: float) -> list[Stream]:
        """Prompts that ended sample their first token, draft and join the rounds; returns those already done."""

        joined, head = [], 0
        for (s, a, n), last in zip(pieces, lasts):
            s.prefill_s += spent
            e, mtp, _, kept = self.fills[s.sid]
            self.fills[s.sid][2] = a + n
            if kept is not None and a < kept[0]["pos"] <= a + n:
                self._remember(list(s.prompt[:kept[0]["pos"]]), s.st, *kept)
            self.fills[s.sid][3] = None                    # only the bounded cache owns a stored snapshot
            if a + n < len(s.prompt):
                continue
            self.filling.remove(s)
            self.fills.pop(s.sid)
            st, e.last_streams = s.st, last
            logits = heads[head:head + 1]
            if s.constraint is not None:                 # a reply's grammar: the first token too
                logits = s.constraint.mask(logits, None, self.w.meta.get("vocab_offset", 0))
            first = e.sample(logits, [len(s.prompt)], s.sampling)[0]
            if s.probabilities is not None:
                capture(logits, [first], [len(s.prompt)], s.probabilities)
            if s.constraint is not None:
                s.constraint.advance([first])
            head += 1
            image_rows.finish(st)
            s.context = list(s.prompt)
            s.drafts = []
            s.copies = Copies(self.depth + 1, self.copy_rows) if mtp and self.copy_rows else None
            s.started = time.perf_counter()
            self.streams[s.sid] = s
            s.take([first], self._ends(s))               # the first token goes out before the next draft
            if mtp and not s.done and s.count > 1:       # the first token in the context: a copy may follow it
                s.drafts = s.copies.propose(s.context, s.count - len(s.out)) if s.copies is not None else []
                if s.drafts:
                    absorb(e, last, [first])
                else:
                    s.drafts = draft(e, last, [first], st.pos + 1, min(self.depth, s.count - 1), s.sampling,
                                     self.confidence)
            if s.done:
                joined.append(s)
        return joined

    def _ends(self, s: Stream) -> tuple[int, ...]:
        """The end tokens that end this stream: none when its request ignores them (``ignore_eos``)."""

        return self.eos if s.stop_eos else ()

    @torch.no_grad()
    def round(self) -> list[Stream]:
        """One round over the live streams, with the next prompt pass in the same forward while prompts fill."""

        ended = self._make_room()                      # every stream's caches hold this round, or the newest wait
        live = [s for s in self.streams.values() if not s.done and not s.waiting]
        if self.filling and (not live or not self.converged):
            ended += self._fill()                      # passes alone; a prompt that ends here joins this round
            live = [s for s in self.streams.values() if not s.done and not s.waiting]
        if not live:
            return ended
        fit(live, self.buf.rows, self.depth + 1)         # before a grammar's window: it masks the rows it keeps
        grammars, failed = {}, []
        for s in live:                                   # a grammar cuts the drafts no accepted path can hold
            if s.constraint is not None:
                tokens = [s.out[-1]] + list(s.drafts)
                try:
                    grammars[s.sid] = s.constraint.window(tokens, list(range(-1, len(tokens) - 1)))
                except GrammarError as exc:              # this request ends with its error, the others go on
                    s.error, s.done = exc, True
                    self.held.pop(s.sid, None)
                    failed.append(s)
                    continue
                s.drafts = grammars[s.sid].tokens[1:]
        live = [s for s in live if not s.done]
        if not live:
            return failed + ended
        t0 = time.perf_counter()
        windows = [(s.st, [s.out[-1]] + list(s.drafts)) for s in live]
        segs = stage(self.w, self.buf, windows)
        # a pass shares the round's forward only where their experts share a launch; else _fill ran it between rounds
        pieces, psegs = (self._pieces(self._pass_rows()) if self.filling and self.converged else []), None
        cuts = []
        if pieces:
            self._note_passed(pieces)
            try:
                self._read_ahead(pieces)
                psegs = stage(self.w, self.pbuf, [(s.st, s.prompt[a:a + n]) for s, a, n in pieces])
                cuts = self._cuts(pieces, psegs)
            except Exception as exc:                     # noqa: BLE001  (the pass's requests fail, the round goes on)
                ended += self._failed(pieces, exc)
                pieces = []
        held = [self.held.pop(s.sid, []) for s in live]
        tables = self.buf.gdn_tables = gdn_multi.Tables(self.w, self.gdn, segs, held)
        self.buf.attn_step = attn_multi.Step(self.w, segs, mtp=False)
        try:
            if pieces:                                 # the window and the pass: each layer's experts once for both
                pends = self._end_rows(pieces, psegs)
                logits, heads = compute_mixed(self.w, segs, self.buf, psegs, self.pbuf, ends=pends, cuts=cuts)
                heads = heads[:len(pends)].clone() if pends else None
            else:
                logits = compute(self.w, segs, self.buf)
        finally:
            self.buf.gdn_tables = self.buf.attn_step = None
        lasts = self._absorb(pieces, psegs, cuts) if pieces else None
        starts = [a0 for _, a0, _ in segs] + [segs[-1][2]]
        for s, (_, a0, a1) in zip(live, segs):
            if s.sid in grammars:
                s.constraint.mask(logits[a0:a1], grammars[s.sid])
        positions = [[st.pos + 1 + r for r in range(a1 - a0)] for st, a0, a1 in segs]
        sampled = sample_streams(logits, starts, positions, [s.sampling for s in live])
        paths = [accept(tokens, list(range(-1, len(tokens) - 1)), rows, s.count - len(s.out), self._ends(s))
                 for s, (_, tokens), rows in zip(live, windows, sampled)]
        for s, (_, tokens), (_, a0, _), (path, end), pos in zip(live, windows, segs, paths, positions):
            if s.probabilities is not None:
                capture(logits, [tokens[r] for r in path[1:]] + [end], [pos[r] for r in path],
                        s.probabilities, rows=[a0 + r for r in path])
        for s, rows in zip(live, gdn_multi.keep(tables, [len(path) for path, _ in paths])):
            self.held[s.sid] = rows                      # the next round's trees fold these rows in first
        kept = []
        for s, (_, tokens), (st, a0, a1), rows, (path, end) in zip(live, windows, segs, sampled, paths):
            commit(self.w, st, self.buf, a1 - a0, len(path), at=a0, states=False)
            s.committed.extend(tokens[:len(path)])
            s.counted(len(tokens))
            new = [tokens[r] for r in path[1:]] + [end]
            if s.constraint is not None:
                try:
                    s.constraint.advance(new)
                except GrammarError as exc:
                    s.error = exc
            last = s.error is not None or len(s.out) + len(new) >= s.count or end in self._ends(s)
            kept.append((s, a0, rows[:len(path)], new, last))
        for s, (_, tokens), (path, _) in zip(live, windows, paths):
            if s.copies is not None:                     # the rows verified: a grammar may have cut some
                s.copies.landed(len(tokens), len(path))
        for s, (_, tokens), (path, _) in zip(live, windows, paths):
            self._landed(s, len(tokens) - 1, len(path) - 1)
        for s, _, _, new, _ in kept:
            if s.error is not None:
                s.done = True
                continue
            s.take(new, self._ends(s))
        # after the take: a copy reads this round's tokens
        self._draft_all([(s, a0, keep) for s, a0, keep, _, last in kept if s.draft and not last])
        spent = time.perf_counter() - t0
        self._timed(spent, sum(n for _, _, n in pieces))
        if not pieces:                                   # a decode round alone: ms a kept token, every stream's
            ms = 1e3 * spent / max(1, sum(len(new) for _, _, _, new, _ in kept))
            self.tok_ms = ms if self.tok_ms is None else 0.8 * self.tok_ms + 0.2 * ms
        if pieces:                                     # prompts that ended in this round's pass join the next
            ended += self._joined(pieces, heads, lasts, spent / len(pieces))
        done = [s for s in live if s.done]
        for s in done:                                   # a finished stream's state is never read again
            self.held.pop(s.sid, None)
        return failed + done + ended

    def _draft_all(self, streams: list) -> None:
        """Every drafting stream absorbs its kept rows, then copies from its context or chains MTP drafts,
        the chains in one step a depth."""

        for s, _, _ in streams:
            s.drafts, s.draft_p, s.chain_q = [], [], 1.0
        room = {s.sid: min(self.depth, s.count - len(s.out)) for s, _, _ in streams}
        todo = [(s, a0, keep) for s, a0, keep in streams if room[s.sid] > 0 and self.mbuf is not None]
        if not todo:
            return
        for s, _, _ in todo:
            st = s.st
            if st.mtp_drafted:
                st.set_mtp_len(st.mtp_len - st.mtp_drafted)
                st.mtp_drafted = 0
        windows = [(s.st, keep, self.buf.streams[a0:a0 + len(keep)]) for s, a0, keep in todo]
        segs = mtp_stage(self.w, self.mbuf, windows)
        logits = self._mtp(segs)
        for (s, _, keep), (st, a0, a1) in zip(todo, segs):
            st.set_mtp_len(st.mtp_len + len(keep))
            s.drafts = s.copies.propose(s.context, s.count - len(s.out)) if s.copies is not None else []
        self._hint([(s, 1 + len(s.drafts)) for s, _, _ in todo])      # while the GPU absorbs: rows 0 and the copies
        chains = [i for i, (s, _, _) in enumerate(todo) if not s.drafts]
        if not chains:
            return
        if len(chains) < len(todo):                      # the logits of the streams that chain
            logits = logits[chains]
        active = [(todo[i][0], segs[i][2] - 1) for i in chains]
        n_live = len(self.streams)
        for j in range(self.depth):
            picks = self._picks(logits, [s.st.pos + 1 + j for s, _ in active], [s.sampling for s, _ in active])
            nxt, added = [], []
            for (s, row), (d, p) in zip(active, picks):
                keep, go = self._chain(s, j, p, room[s.sid], len(active), n_live)
                if not keep:
                    continue
                s.drafts.append(d)
                s.draft_p.append(p)
                added.append((s, 1))
                if go:
                    nxt.append((s, row, d))
            if not nxt:
                self._hint(added)
                return
            windows = [(s.st, [d], self.mbuf.streams[row:row + 1]) for s, row, d in nxt]
            segs = mtp_stage(self.w, self.mbuf, windows)
            logits = self._mtp(segs)
            self._hint(added)                                # while the GPU runs the next step
            for s, _, _ in nxt:
                s.st.set_mtp_len(s.st.mtp_len + 1)
                s.st.mtp_drafted += 1
            active = [(s, a0) for (s, _, _), (_, a0, _) in zip(nxt, segs)]

    def _chain(self, s: Stream, j: int, p: float, room: int, chaining: int, live: int) -> tuple[bool, bool]:
        """Chain step ``j``'s draft of probability ``p``: (keep it, chain one more). The first draft is always kept,
        as in v0.6.1; a draft a rule drops ends its chain."""

        rule = self.chain_rule
        if rule == "conf":
            low = self.confidence > 0 and p < self.confidence
            return not (low and j > 0), not low and j + 1 < room
        c = self._cal(p) * (getattr(s, "trust", 1.0) if rule in ("adapt", "acum") else 1.0)
        q = s.chain_q * min(1.0, c)                      # the chance this draft and every earlier one land
        if rule in ("cum", "acum"):
            keep = j == 0 or q >= self.chain_t
            if keep:
                s.chain_q = q
            return keep, keep and q >= self.chain_t and j + 1 < room
        tok = self.tok_ms
        if tok is None:                                  # no decode round timed yet: the confidence rule
            low = self.confidence > 0 and p < self.confidence
            return not (low and j > 0), not low and j + 1 < room
        f = min(1.0, max(0.0, (live - 1) / 3))
        row = ROW_MS[0] + (ROW_MS[1] - ROW_MS[0]) * f   # a verify row, interpolated by the live streams
        keep = j == 0 or q * tok >= row
        if keep:
            s.chain_q = q
        step = MTP_MS / max(1, chaining)                 # the next MTP step, shared by the streams still chaining
        return keep, keep and j + 1 < room and q * min(1.0, c) * tok >= step + row

    @staticmethod
    def _cal(p: float) -> float:
        """A draft's chance of landing, given every earlier draft landed, by its draft-head probability (``CAL``)."""

        if p <= CAL[0][0]:
            return CAL[0][1]
        for (p0, r0), (p1, r1) in zip(CAL, CAL[1:]):
            if p <= p1:
                return r0 + (r1 - r0) * (p - p0) / (p1 - p0)
        return CAL[-1][1]

    def _landed(self, s: Stream, drafted: int, landed: int) -> None:
        """After a round's verify: ``landed`` of the ``drafted`` rows held. A chain's drafts tune the stream's trust
        (its landings over their calibrated chances, drafts up to the first miss); the probe sees every verdict."""

        probs = getattr(s, "draft_p", [])[:drafted]
        if not probs:
            return
        if self.verdict is not None:
            self.verdict(s, list(probs), landed)
        seen = probs[:landed + 1]
        want = sum(self._cal(p) for p in seen)
        if want > 0:
            ratio = min(1.5, max(0.5, min(landed, len(seen)) / want))
            s.trust = 0.8 * getattr(s, "trust", 1.0) + 0.2 * ratio

    def _hint(self, rows: list) -> None:
        """Start the page reads of each (stream, last) pair's next-round rows: its window so far, the ``last`` final
        rows (on the hint thread; the stage reads the same bytes either way)."""

        if self.hints is not None and rows:
            self.hints.submit(_will_need, self.w,
                              [(s.st.ple_history, [s.out[-1], *s.drafts], last) for s, last in rows
                               if s.st.ple_history is not None])

    def _mtp(self, segs: list) -> torch.Tensor:
        """An MTP step over every drafting stream, its attention one launch a kernel for all of them."""

        self.mbuf.attn_step = attn_multi.Step(self.w, segs, mtp=True)
        try:
            return mtp_compute(self.w, segs, self.mbuf)
        finally:
            self.mbuf.attn_step = None

    def _picks(self, logits: torch.Tensor, positions: list[int], samplings: list) -> list[tuple[int, float]]:
        """Each row's keyed draft and its probability at temperature 1, one read-back (drafts change speed only)."""

        row = logits.float()
        k = max([int(s.top_k) + MARGIN for s in samplings if s is not None and s.temperature > 0 and s.top_k] or [1])
        k = min(k, row.shape[1])
        vals, idx = torch.topk(row, k, dim=-1, sorted=False)
        top, col = row.max(dim=-1, keepdim=True)
        lse = torch.logsumexp(row, dim=-1, keepdim=True)
        got = torch.cat([vals, idx.float(), top, col.float(), lse], dim=1).cpu().numpy()
        out = []
        for i, (pos, smp) in enumerate(zip(positions, samplings)):
            g = got[i]
            lse_i = float(g[2 * k + 2])
            if smp is None or smp.temperature <= 0:
                c = int(g[2 * k + 1])
                out.append((int(self.draft_host[c]) if self.draft_host is not None else c,
                            float(np.exp(float(g[2 * k]) - lse_i))))
                continue
            cols = g[k:2 * k].astype(np.int64)
            ids = self.draft_host[cols] if self.draft_host is not None else cols
            tok = choose_rows(g[None, :k].astype(np.float32), ids[None, :], [pos], smp)[0]
            hit = np.nonzero(ids == tok)[0]
            out.append((int(tok), float(np.exp(float(g[hit[0]]) - lse_i)) if len(hit) else 0.0))
        return out

    def finish(self, done: list[Stream]) -> None:
        """Drop finished streams; a slot whose prompt state is kept stays with it, the rest are free again."""

        for s in done:
            self.streams.pop(s.sid, None)
            if not any(k[1] is s.st for k in self.kept) and all(f is not s.st for f in self.free):
                self._shrink(s.st)
                self.free.append(s.st)

    def drop(self) -> list[Stream]:
        live = [s for s in self.streams.values() if not s.done] + self.filling
        self.filling, self.fills = [], {}
        for s in live:
            self.streams.pop(s.sid, None)
            self.held.pop(s.sid, None)
            self._drop_kept(s.st)
            self._shrink(s.st)
            self.free.append(s.st)
        return live


def _tensors(st: State):
    for value in vars(st).values():
        for v in value if isinstance(value, list) else [value]:
            if isinstance(v, torch.Tensor):
                yield v
            elif hasattr(v, "__dict__"):                  # scratch and KV cache objects, the MTP head's too
                yield from (t for t in vars(v).values() if isinstance(t, torch.Tensor))
