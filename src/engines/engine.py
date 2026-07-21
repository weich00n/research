"""Simulation engine (VacSim: engines/engine.py).

Runs the CLAUDE.md weekly loop over conditions C0-C3. Each timestep (1 week),
for each agent:
1. Receive policy news (C2/C3).
2. Read social posts from followed agents, posted at t-1 (C1/C3).
3. Convert inputs into fertility-relevant memories (perception LLM call),
   each scored for TPB relevance.
4-6. Retrieve memories: saliency decay + TPB relevance, <=5 seed + <=5 per
   construct, max 20 (lesson.retrieve_memories).
7. Two independent LLM calls: (7a) update TPB scores + reflection, and (7b)
   update the fertility-intention distribution WITHOUT seeing the TPB scores,
   so the TPB->intention link is measured (mediation), not instructed.
8. Agent may post a fertility-related message (visible to followers at t+1).
9. State saved after every timestep.
"""

import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from sandbox.lesson import Lesson, retrieve_memories
from utils.logging_utils import setup_logger
from sandbox.prompts import (
    build_baseline_intention_prompt,
    build_baseline_tpb_prompt,
    build_batch_perception_prompt,
    build_intention_update_prompt,
    build_perception_prompt,
    build_seed_memory_prompt,
    build_tpb_update_prompt,
    build_tweet_prompt,
)

# condition -> (policy news on, social posts on)
CONDITIONS = {
    "C0": (False, False),  # static baseline
    "C1": (False, True),   # social only
    "C2": (True, False),   # policy only
    "C3": (True, True),    # policy + social
}

# reflections are not separately rated by the LLM (CLAUDE.md belief-update
# output has no importance field), so they get a fixed moderate importance
REFLECTION_IMPORTANCE = 0.6


class Simulation:
    """Drives the weekly loop for one experimental condition (C0-C3).

    Holds the agents, the social network, the LLM client, the relevance
    `scorer`, and (for policy conditions) the news schedule. `run()` is the
    entry point; `step()` does one week; `save()` checkpoints to JSON.
    """

    def __init__(self, agents, network, condition, llm, scorer,
                 news_schedule=None, output_dir=os.path.join("outputs", "runs"),
                 run_name=None, verbose=True, concurrency=32, rerank_top_k=12,
                 gate_no_input=True):
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {list(CONDITIONS)}")
        self.agents = agents
        self.agents_by_id = {a.agent_id: a for a in agents}
        self.network = network  # {agent_id: [followed agent_ids]}
        self.condition = condition
        self.llm = llm
        self.scorer = scorer  # LLM_judge.RelevanceScorer
        # hybrid-mode cosine shortlist size per construct (passed to scorer.rerank)
        self.rerank_top_k = rerank_top_k
        self.news_schedule = news_schedule or {}
        # Update gating: belief revision requires new evidence. When an agent
        # receives no new inputs in a week (no policy news, no friend posts),
        # skip its LLM update calls and carry the previous scores forward —
        # the news sanity eval measured ~+0.05/week phantom attitude drift
        # from updating on an unchanged information set (no-news control cell,
        # outputs/analysis/news_corpus_results.md §3b). Disable with
        # gate_no_input=False (driver --no-gating) to reproduce old behaviour.
        self.gate_no_input = gate_no_input
        self.output_dir = output_dir
        self.run_name = run_name or f"run_{condition}"
        self.verbose = verbose
        # How many agents' weeks run concurrently. Each agent's LLM calls are
        # independent across agents (only its own state is mutated), so the
        # weekly loop fans out over a thread pool; vLLM's continuous batching
        # turns these concurrent requests into GPU throughput.
        self.concurrency = concurrency
        self.current_timestep = 0

        # console shows progress; <run_name>.log also captures DEBUG detail
        # (every LLM call, retries, raw responses that failed JSON parsing)
        self.logger = setup_logger(
            log_path=os.path.join(output_dir, f"{self.run_name}.log"),
            console_level=logging.INFO if verbose else logging.WARNING,
        )
        self.logger.info(f"Simulation: condition={condition}, "
                         f"agents={len(agents)}, run_name={self.run_name}, "
                         f"relevance_mode={getattr(scorer, 'mode', None)}")

        policy_on, _ = CONDITIONS[condition]
        if policy_on and not self.news_schedule:
            raise ValueError(f"{condition} needs a news_schedule (see news.build_news_schedule)")

    def _log(self, msg):
        self.logger.info(msg)

    # ── Initialisation ─────────────────────────────────────────────────────

    def initialise_seed_memories(self):
        """Generate 5 profile seed memories per agent (skips agents that have them)."""
        for agent in self.agents:
            if agent.seed_lessons:
                continue
            system, user = build_seed_memory_prompt(agent)
            out = self.llm.chat_json(system, user)
            # Asked for {"memories": [...]}, but the model occasionally drops the
            # wrapper and returns the bare array. Accept either shape rather than
            # crashing a long init mid-run on one off-format response.
            memories = out.get("memories", out) if isinstance(out, dict) else out
            if not isinstance(memories, list):
                raise ValueError(
                    f"{agent.agent_id}: seed-memory response not a list of memories: {out!r}")
            for mem in memories[:5]:
                rel, cos = self.scorer.creation_scores(mem["memory_text"])
                lesson = Lesson(
                    agent_id=agent.agent_id,
                    memory_text=mem["memory_text"],
                    created_timestep=0,
                    source_type="profile_seed",
                    importance=mem.get("importance", 0.5),
                    memory_class="seed",
                    relevance=rel,
                    cosine_relevance=cos,
                )
                agent.add_lesson(lesson)
            self._log(f"{agent.agent_id}: {len(agent.seed_lessons)} seed memories")
            # Checkpoint after every agent so a crash/stall during the long seed
            # phase costs at most one agent (the whole world state is rewritten;
            # current_timestep stays 0 = "seeds only, no week run yet").
            self.save()

    def _run_agent_baseline(self, agent):
        """Establish one agent's t=0 belief state from its seed memories.

        COMPUTE then COMMIT, like `_run_agent_week`: the two LLM calls run first,
        and the agent's state is mutated only afterwards, so a failure leaves the
        agent untouched and retriable (idempotent). Deterministic (temperature 0)
        so the baseline is reproducible. Writes scores only — no reflection memory
        and no tweet, so the t=0 memory stream stays the pure seed memories.
        """
        # Only seed memories exist at t=0, so retrieval returns the <=5 seeds.
        retrieved, _ = retrieve_memories(agent.lessons, 0)
        sys_t, usr_t = build_baseline_tpb_prompt(agent, retrieved)
        tpb_out = self._chat_json(sys_t, usr_t, temperature=0.0)
        sys_i, usr_i = build_baseline_intention_prompt(agent, retrieved)
        int_out = self._chat_json(sys_i, usr_i, temperature=0.0)

        # ── COMMIT (no LLM calls below) ─────────────────────────────────────
        agent.update_belief_state(
            tpb_out["attitude_score"],
            tpb_out["subjective_norm_score"],
            tpb_out["pbc_score"],
            int_out["fertility_intention"],
            timestep=0,
        )
        self._log(f"baseline {agent.agent_id}: "
                  f"att={agent.belief_state['attitude_score']:.1f} "
                  f"norm={agent.belief_state['subjective_norm_score']:.1f} "
                  f"pbc={agent.belief_state['pbc_score']:.1f} "
                  f"intent={agent.belief_state['fertility_intention_dist']}")

    def initialise_baseline(self):
        """Set every fresh agent's grounded t=0 belief state from its seed memories.

        Idempotent: agents that already have a belief_history (resumed, mid-run, or
        already baselined) are skipped, so this is a cheap no-op when a pre-seeded
        agent file is loaded. Fans the fresh agents out over the thread pool with
        the same initial-pass + up-to-2-retry structure as `step()`. Does not save;
        the caller (`run()` / `--init-only`) persists the result.
        """
        pending = [a for a in self.agents if not a.belief_history]
        if not pending:
            return
        self._log(f"Establishing t=0 baseline for {len(pending)} agents")
        for attempt in range(3):
            failed = []
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = {pool.submit(self._run_agent_baseline, a): a for a in pending}
                for fut in as_completed(futures):
                    agent = futures[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        self.logger.warning(f"baseline {agent.agent_id}: failed: {e}")
                        failed.append(agent)
            pending = failed
            if not pending:
                break
            self.logger.warning(f"baseline: {len(pending)} agents failed on attempt "
                                f"{attempt + 1}, retrying: {[a.agent_id for a in pending]}")
        if pending:
            raise RuntimeError(
                f"baseline: {len(pending)} agents still failing after retries "
                f"({[a.agent_id for a in pending]})")

    # ── One timestep ───────────────────────────────────────────────────────

    def _chat_json(self, system, user, temperature=None):
        """chat_json against a randomly-picked endpoint (spreads load when several
        local vLLM servers are configured via LOCAL_LLM_URLS; a no-op for one URL).

        `temperature=None` leaves the client's default (weekly calls); the baseline
        pass passes 0.0 for deterministic, reproducible initialisation.
        """
        url = random.choice(self.llm.urls) if len(self.llm.urls) > 1 else None
        return self.llm.chat_json(system, user, url=url, temperature=temperature)

    def _build_perceived(self, agent, message_text, message_kind, source_type,
                         timestep, source_agent_id=None):
        """Steps 3+5: turn an incoming message into a relevance-scored memory.

        Builds (but does NOT attach) the Lesson, so the caller can defer all
        state mutation until every LLM call in the week has succeeded.
        """
        system, user = build_perception_prompt(agent, message_text, message_kind)
        out = self._chat_json(system, user)
        rel, cos = self.scorer.creation_scores(out["memory_text"])
        return Lesson(
            agent_id=agent.agent_id,
            memory_text=out["memory_text"],
            created_timestep=timestep,
            source_type=source_type,
            importance=out.get("importance", 0.5),
            memory_class="simulation",
            source_agent_id=source_agent_id,
            relevance=rel,
            cosine_relevance=cos,
        )

    def _build_perceived_batch(self, agent, items, message_kind, source_type, timestep):
        """Batched sibling of _build_perceived: turns a WEEK's worth of
        same-kind messages into distinct, deduplicated relevance-scored
        memories in ONE call (VacSim's feed_tweets() -> generate_and_save_
        lessons() pattern), instead of one call per message. Several
        followed agents posting similar content in the same week can then
        collapse into a single takeaway instead of each independently
        filling a construct's retrieval slots.

        `items` is [(source_agent_id, text), ...]. Returns a list of Lessons
        (not attached — same COMPUTE-then-COMMIT contract as _build_perceived).
        A memory whose source_indices resolve to exactly one input item keeps
        that item's source_agent_id; a genuine cross-item merge gets
        source_agent_id=None (no analysis code keys off this field today, so
        losing provenance on the rare merged case is safe).
        """
        numbered = list(enumerate((text for _, text in items), start=1))
        system, user = build_batch_perception_prompt(agent, numbered, message_kind)
        out = self._chat_json(system, user)
        # Same tolerant-of-a-bare-list handling as initialise_seed_memories,
        # for the same reason: don't crash a long run on one off-format reply.
        memories = out.get("memories", out) if isinstance(out, dict) else out
        if not isinstance(memories, list):
            raise ValueError(
                f"{agent.agent_id}: batch perception response not a list "
                f"of memories: {out!r}")
        lessons = []
        for mem in memories:
            indices = mem.get("source_indices") or []
            source_agent_id = items[indices[0] - 1][0] if len(indices) == 1 else None
            rel, cos = self.scorer.creation_scores(mem["memory_text"])
            lessons.append(Lesson(
                agent_id=agent.agent_id,
                memory_text=mem["memory_text"],
                created_timestep=timestep,
                source_type=source_type,
                importance=mem.get("importance", 0.5),
                memory_class="simulation",
                source_agent_id=source_agent_id,
                relevance=rel,
                cosine_relevance=cos,
            ))
        return lessons

    def _run_agent_week(self, agent, timestep, news_items):
        """Run one CLAUDE.md week (steps 1-8) for a single agent.

        COMPUTE then COMMIT: every LLM call runs first, building local objects;
        the agent's own state is mutated only at the very end. So if any call
        raises, nothing was committed and the agent can be retried cleanly
        (idempotent) — no duplicate memories, no rollback.

        Safe to run concurrently across agents: it mutates only `agent`, reads
        other agents' tweets read-only (they were posted at t-1), and the Lesson/
        Tweet id counters are atomic under the GIL.
        """
        _, social_on = CONDITIONS[self.condition]
        new_lessons = []   # perceived memories, attached on commit
        new_messages = []

        # 1. policy news
        for news in news_items:
            new_lessons.append(
                self._build_perceived(agent, news.text, "news report", "policy_news", timestep))
            new_messages.append(f"[policy news] {news.text}")

        # 2. social posts from followed agents, posted at t-1 — batched into
        # ONE perception call for the week (see _build_perceived_batch),
        # instead of one call per tweet, so several friends posting similar
        # content collapses into one takeaway rather than each independently
        # flooding a construct's retrieval slots.
        if social_on:
            tweet_items = []
            for followed_id in self.network.get(agent.agent_id, []):
                followed = self.agents_by_id[followed_id]
                for tweet in followed.tweets:
                    if tweet.created_timestep == timestep - 1:
                        tweet_items.append((followed_id, tweet.text))
                        new_messages.append(f"[post by a friend] {tweet.text}")
            if tweet_items:
                new_lessons.extend(self._build_perceived_batch(
                    agent, tweet_items, "social media posts", "social_post", timestep))

        # Gate: no new inputs this week -> no update. Carry the previous
        # belief state forward as this week's belief_history entry (keeps
        # trajectories week-aligned) and skip every LLM call: no retrieval,
        # no TPB/intention update, no reflection, no post.
        #
        # Exception: week 1 of a social-on condition is never gated, even if
        # this agent has no friend posts yet (nobody can, on week 1 -- no one
        # has posted before). Posting only happens past the gate (step 8), so
        # without this exception a social-only run deadlocks permanently: no
        # one can post in week 1 -> no one has anything to read in week 2 ->
        # no one can post in week 2 -> ... (found via run_C1_Qwen, 2026-07-20:
        # 0/100 agents moved across 12 weeks). This unblocks exactly that
        # first week; C2/C3 are unaffected since policy news already keeps
        # the gate open in week 1 regardless.
        if self.gate_no_input and not new_lessons and not (social_on and timestep == 1):
            b = agent.belief_state
            agent.update_belief_state(
                b["attitude_score"], b["subjective_norm_score"], b["pbc_score"],
                b["fertility_intention_dist"], timestep)
            self._log(f"t={timestep} {agent.agent_id}: no new inputs, "
                      f"gated (scores carried forward)")
            return

        # 4-6. retrieval over existing + this week's freshly-perceived memories.
        # In hybrid mode, rerank() first cosine-shortlists candidates and LLM-judges
        # them (filling lesson.relevance, cached); a no-op in llm/cosine modes.
        # (On a retried agent this re-runs; rerank is idempotent via its cache and
        # retrieval may re-append to retrieval_history — analytics-only, harmless.)
        all_lessons = agent.lessons + new_lessons
        self.scorer.rerank(all_lessons, top_k=self.rerank_top_k)
        retrieved, construct_map = retrieve_memories(all_lessons, timestep)

        # 7a. TPB update (attitude/norm/pbc + reflection) — intention NOT here.
        sys_t, usr_t = build_tpb_update_prompt(agent, retrieved, new_messages, timestep)
        tpb_out = self._chat_json(sys_t, usr_t)
        # 7b. Fertility intention — generated independently, WITHOUT seeing the
        #     numeric TPB scores, so the TPB->intention link is measured
        #     (mediation), not instructed. Reads the agent's *previous* state,
        #     which is still intact because commit happens below.
        sys_i, usr_i = build_intention_update_prompt(agent, retrieved, new_messages, timestep)
        int_out = self._chat_json(sys_i, usr_i)

        reflection_text = tpb_out.get("reflection_memory") or ""
        reflection_lesson = None
        if reflection_text:
            rel, cos = self.scorer.creation_scores(reflection_text)
            reflection_lesson = Lesson(
                agent_id=agent.agent_id,
                memory_text=reflection_text,
                created_timestep=timestep,
                source_type="reflection",
                importance=REFLECTION_IMPORTANCE,
                memory_class="simulation",
                relevance=rel,
                cosine_relevance=cos,
            )

        # 8. optional social post (visible to followers at t+1)
        tweet_text = None
        if social_on:
            system, user = build_tweet_prompt(agent, reflection_text, timestep)
            tweet_out = self._chat_json(system, user)
            if tweet_out.get("post") and tweet_out.get("text"):
                tweet_text = tweet_out["text"]

        # ── COMMIT (no LLM calls below — cannot fail, so state stays consistent) ─
        for lesson in new_lessons:
            agent.add_lesson(lesson)
        for lesson in retrieved:
            lesson.used_in_update = True
        agent.update_belief_state(
            tpb_out["attitude_score"],
            tpb_out["subjective_norm_score"],
            tpb_out["pbc_score"],
            int_out["fertility_intention"],
            timestep,
        )
        if reflection_lesson:
            agent.add_lesson(reflection_lesson)
        if tweet_text:
            agent.post_tweet(tweet_text, timestep)

        self.logger.debug(f"t={timestep} {agent.agent_id}: read {len(new_messages)} "
                          f"messages, retrieved {len(retrieved)} memories {construct_map}")
        self._log(f"t={timestep} {agent.agent_id}: "
                  f"att={agent.belief_state['attitude_score']:.1f} "
                  f"norm={agent.belief_state['subjective_norm_score']:.1f} "
                  f"pbc={agent.belief_state['pbc_score']:.1f}")

    def _run_week_batch(self, agents, timestep, news_items):
        """Run one week for each agent concurrently; return the agents that failed."""
        failed = []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(self._run_agent_week, a, timestep, news_items): a
                       for a in agents}
            for fut in as_completed(futures):
                agent = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    self.logger.warning(f"t={timestep} {agent.agent_id}: week failed: {e}")
                    failed.append(agent)
        return failed

    def step(self, timestep):
        """Run one simulation week for every agent (CLAUDE.md steps 1-9).

        Agents are processed concurrently (`concurrency` at a time). Each agent
        perceives policy news (1) and friends' t-1 posts (2), retrieves the most
        relevant memories (4-6), gets new TPB scores + reflection (7a) and — in a
        separate call that never sees those scores — its fertility intention (7b),
        and may post a tweet (8). Agents whose week raises (after the client's own
        retries) are retried on their own; if any still fail the week is aborted
        with NO partial save, so --resume restarts it cleanly. The whole world
        state is saved once at the end of the week (9).
        """
        self.current_timestep = timestep
        step_start = time.time()
        policy_on, _ = CONDITIONS[self.condition]
        news_items = self.news_schedule.get(timestep, []) if policy_on else []
        if news_items:
            self.logger.debug(f"t={timestep} news: "
                              f"{[n.policy.name for n in news_items]}")

        # Initial pass + up to 2 retries of just the stragglers.
        pending = list(self.agents)
        for attempt in range(3):
            pending = self._run_week_batch(pending, timestep, news_items)
            if not pending:
                break
            self.logger.warning(f"t={timestep}: {len(pending)} agents failed on "
                                f"attempt {attempt + 1}, retrying: "
                                f"{[a.agent_id for a in pending]}")
        if pending:
            raise RuntimeError(
                f"t={timestep}: {len(pending)} agents still failing after retries "
                f"({[a.agent_id for a in pending]}); aborting week (no save) so "
                f"--resume restarts it from the previous checkpoint.")

        # 9. save after every timestep so long runs are resumable/inspectable
        path = self.save()
        self.logger.info(f"Week {timestep} done in {time.time() - step_start:.0f}s, "
                         f"state saved to {path}")

    def run(self, num_timesteps, start_timestep=1):
        """Initialise seed memories (once, if missing) then step `num_timesteps` weeks."""
        # Seed memories are the week-0 background beliefs; only generated if the
        # agents don't already have them (so re-runs / resumes skip the LLM cost).
        if not all(a.seed_lessons for a in self.agents):
            self.initialise_seed_memories()
            self.save()
        # t=0 grounded baseline from the seeds (idempotent: a no-op when the agents
        # were loaded pre-baselined from agents_final_100_seeded.json).
        self.initialise_baseline()
        self.save()
        for t in range(start_timestep, start_timestep + num_timesteps):
            self._log(f"\n===== Week {t} ({self.condition}) =====")
            self.step(t)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self):
        """Write the full world state (condition, week, network, all agents) to JSON.

        Called after every week, so the latest <run_name>.json is always a
        complete, inspectable snapshot of the run.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"{self.run_name}.json")
        state = {
            "condition": self.condition,
            "current_timestep": self.current_timestep,
            "network": self.network,
            "agents": [a.to_dict() for a in self.agents],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        return path
