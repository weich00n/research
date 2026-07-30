"""VacSim-direct replication engine: no TPB layer (~/VacSim: engines/engine.py).

Strips the TPB belief-mediation architecture out entirely and instead follows
VacSim's own mechanics on the same seeded 100-agent pool:

- Memories are formed in WEEKLY BATCHES: all of a week's news compiled into
  ONE prompt/call (VacSim's broadcast_news_and_policies), and all of a week's
  followed-agent posts compiled into ONE prompt/call (VacSim's feed_tweets) --
  not one perception call per item like engines.engine.
- Retrieval scores memories by importance + time decay only (VacSim's
  Lesson.score), no per-construct relevance tagging, one pooled ranking (no
  seed/simulation split).
- The fertility_intention distribution is updated by a SINGLE call per week
  (VacSim's attitude_prompt) that sees the top-k retrieved memories -- there
  is no TPB layer to decouple it from.

This is a separate methodology from engines.engine (the TPB mediation study):
it does not import or mutate that engine's Agent schema, retrieval formula, or
belief-update calls. Intended to run ALONGSIDE C0-C3 from engines.engine for
comparison (e.g. to check whether TPB-side score drift was leaking into the
mediation study's intention trajectories via the reflection-memory path), not
to replace them.
"""

import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from engines.engine import CONDITIONS
from sandbox.lesson import LAMBDA, Lesson
from sandbox.prompts import build_seed_memory_prompt
from sandbox.prompts_direct import (
    MAX_LESSONS_PER_BATCH,
    build_baseline_intention_prompt,
    build_intention_update_prompt,
    build_news_lessons_prompt,
    build_social_lessons_prompt,
    build_tweet_prompt,
)
from utils.logging_utils import setup_logger

# VacSim's own retrieval knobs (sandbox/agent.py: retrieve_reflections /
# max_reflections=5, and the score>0.05 filter before normalising).
RETRIEVE_TOP_K = 5
MIN_RETRIEVE_SCORE = 0.05


def _vacsim_score(lesson, current_timestep):
    """VacSim's Lesson.score: importance + decay^age (additive, not
    multiplicative, and with no relevance term at all -- unlike
    sandbox.lesson.Lesson.saliency, which this methodology does not use)."""
    return lesson.importance + LAMBDA ** (current_timestep - lesson.created_timestep)


def retrieve_top_k(lessons, current_timestep, k=RETRIEVE_TOP_K, min_score=MIN_RETRIEVE_SCORE):
    """VacSim's retrieve_reflections: filter out near-zero scores, min-max
    normalise what's left, take the top k. Pools seed + simulation memories
    together in one ranking -- VacSim keeps all lessons in a single set."""
    scored = [(l, _vacsim_score(l, current_timestep)) for l in lessons]
    scored = [(l, s) for l, s in scored if s > min_score]
    if not scored:
        return []
    scores = [s for _, s in scored]
    lo, hi = min(scores), max(scores)
    if hi > lo:
        scored = [(l, (s - lo) / (hi - lo)) for l, s in scored]
    else:
        scored = [(l, 1.0) for l, _ in scored]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [l for l, _ in scored[:k]]


def _parse_lessons(out, agent_id, timestep, source_type, max_lessons=MAX_LESSONS_PER_BATCH):
    """Parse a batched lesson-generation response into Lesson objects.

    Asked for {"lessons": [...]}, but tolerate a bare list like the perception
    prompts elsewhere in this project do.
    """
    items = out.get("lessons", out) if isinstance(out, dict) else out
    if not isinstance(items, list):
        return []
    lessons = []
    for item in items[:max_lessons]:
        text = item.get("memory_text") if isinstance(item, dict) else None
        if not text:
            continue
        lessons.append(Lesson(
            agent_id=agent_id,
            memory_text=text,
            created_timestep=timestep,
            source_type=source_type,
            importance=item.get("importance", 0.5),
            memory_class="simulation",
        ))
    return lessons


class DirectSimulation:
    """Weekly loop for the VacSim-direct (no-TPB) replication, condition C0-C3.

    Holds the agents (sandbox.agent_direct.DirectAgent), the social network,
    the LLM client, and (for policy conditions) the news schedule. `run()` is
    the entry point; `step()` does one week; `save()` checkpoints to JSON.
    """

    def __init__(self, agents, network, condition, llm,
                 news_schedule=None, output_dir=os.path.join("outputs", "runs"),
                 run_name=None, verbose=True, concurrency=32, gate_no_input=True):
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {list(CONDITIONS)}")
        self.agents = agents
        self.agents_by_id = {a.agent_id: a for a in agents}
        self.network = network  # {agent_id: [followed agent_ids]}
        self.condition = condition
        self.llm = llm
        self.news_schedule = news_schedule or {}
        # Same anti-phantom-drift rationale as engines.engine.Simulation: skip
        # the update call in weeks with no new inputs, carrying the previous
        # intention forward instead of re-asking the LLM to judge an unchanged
        # information set. Disable with gate_no_input=False for old behaviour.
        self.gate_no_input = gate_no_input
        self.output_dir = output_dir
        self.run_name = run_name or f"run_{condition}_direct"
        self.verbose = verbose
        self.concurrency = concurrency
        self.current_timestep = 0

        self.logger = setup_logger(
            log_path=os.path.join(output_dir, f"{self.run_name}.log"),
            console_level=logging.INFO if verbose else logging.WARNING,
        )
        self.logger.info(f"DirectSimulation (no-TPB): condition={condition}, "
                         f"agents={len(agents)}, run_name={self.run_name}")

        policy_on, _ = CONDITIONS[condition]
        if policy_on and not self.news_schedule:
            raise ValueError(f"{condition} needs a news_schedule "
                            f"(see sandbox.news.build_news_schedule)")

    def _log(self, msg):
        self.logger.info(msg)

    def _chat_json(self, system, user, temperature=None):
        url = random.choice(self.llm.urls) if len(self.llm.urls) > 1 else None
        return self.llm.chat_json(system, user, url=url, temperature=temperature)

    # ── Initialisation ─────────────────────────────────────────────────────

    def initialise_seed_memories(self):
        """Generate 5 profile seed memories per agent (skips agents that have
        them). Reuses the TPB study's own seed-memory prompt since it is
        profile-only with no TPB framing. Normally a no-op here: agents loaded
        from agents_final_100_seeded.json already carry these seeds.
        """
        for agent in self.agents:
            if agent.seed_lessons:
                continue
            system, user = build_seed_memory_prompt(agent)
            out = self._chat_json(system, user)
            memories = out.get("memories", out) if isinstance(out, dict) else out
            if not isinstance(memories, list):
                raise ValueError(
                    f"{agent.agent_id}: seed-memory response not a list of memories: {out!r}")
            for mem in memories[:5]:
                agent.add_lesson(Lesson(
                    agent_id=agent.agent_id,
                    memory_text=mem["memory_text"],
                    created_timestep=0,
                    source_type="profile_seed",
                    importance=mem.get("importance", 0.5),
                    memory_class="seed",
                ))
            self._log(f"{agent.agent_id}: {len(agent.seed_lessons)} seed memories")
            self.save()

    def _run_agent_baseline(self, agent):
        """Establish one agent's t=0 fertility_intention from its seed memories
        (VacSim's init_agents). Deterministic (temperature 0) for reproducibility.
        """
        retrieved = retrieve_top_k(agent.lessons, 0)
        sys_i, usr_i = build_baseline_intention_prompt(agent, retrieved)
        out = self._chat_json(sys_i, usr_i, temperature=0.0)
        agent.update_intention(out["fertility_intention"], 0, out.get("reasoning", ""))
        self._log(f"baseline {agent.agent_id}: intent={agent.fertility_intention_dist}")

    def initialise_baseline(self):
        """Set every fresh agent's t=0 fertility_intention. Idempotent: agents
        that already have intention_history (resumed, mid-run, or already
        baselined) are skipped."""
        pending = [a for a in self.agents if not a.intention_history]
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

    def _gather_news_lessons(self, agent, timestep, news_items):
        """Step 1: policy news -> lessons. ALL of this week's news batched into
        ONE call (VacSim's broadcast_news_and_policies), not one call per
        article. Broadcast: every agent reads the same fixed weekly item(s)
        (`self.news_schedule`). Overridden by RecsysDirectSimulation to
        recommend a personalised subset instead."""
        if not news_items:
            return []
        news_texts = [n.text for n in news_items]
        sys_n, usr_n = build_news_lessons_prompt(agent, news_texts)
        out = self._chat_json(sys_n, usr_n)
        return _parse_lessons(out, agent.agent_id, timestep, "policy_news")

    def _gather_social_lessons(self, agent, timestep, social_on):
        """Step 2: followed-agent posts -> lessons. ALL of this week's posts
        batched into ONE call (VacSim's feed_tweets), not one call per tweet.
        Feed: every post from a followed agent at t-1, unranked. Overridden by
        RecsysDirectSimulation to recommend a personalised, similarity-ranked
        subset from the whole tweet pool instead."""
        if not social_on:
            return []
        post_texts = []
        for followed_id in self.network.get(agent.agent_id, []):
            followed = self.agents_by_id[followed_id]
            for tweet in followed.tweets:
                if tweet.created_timestep == timestep - 1:
                    post_texts.append(tweet.text)
        if not post_texts:
            return []
        sys_s, usr_s = build_social_lessons_prompt(agent, post_texts)
        out = self._chat_json(sys_s, usr_s)
        return _parse_lessons(out, agent.agent_id, timestep, "social_post")

    def _run_agent_week(self, agent, timestep, news_items):
        """Run one week for a single agent: news lesson-forming (1), social
        lesson-forming (2), gate, VacSim-style retrieval (4-6), a single
        intention-update call (7), and an optional post (8).

        COMPUTE then COMMIT, like engines.engine: every LLM call runs first;
        the agent's own state is mutated only at the end, so a failure leaves
        the agent untouched and cleanly retriable.
        """
        _, social_on = CONDITIONS[self.condition]
        new_lessons = (self._gather_news_lessons(agent, timestep, news_items)
                       + self._gather_social_lessons(agent, timestep, social_on))

        # Gate: no new inputs this week -> no update; carry the previous
        # intention forward. Week 1 of a social-on condition is never gated
        # (nobody has posted yet, so no one would ever unblock otherwise --
        # same fix as engines.engine._run_agent_week).
        if self.gate_no_input and not new_lessons and not (social_on and timestep == 1):
            agent.intention_history.append({
                "timestep": timestep,
                "fertility_intention_dist": agent.fertility_intention_dist,
                "reasoning": "(gated: no new inputs this week)",
            })
            self._log(f"t={timestep} {agent.agent_id}: no new inputs, gated "
                      f"(intention carried forward)")
            return

        # 4-6. VacSim-style retrieval: importance + decay only, one pooled
        # ranking (no TPB constructs, no seed/simulation split).
        all_lessons = agent.lessons + new_lessons
        retrieved = retrieve_top_k(all_lessons, timestep)

        # 7. single call: fertility intention update. No TPB scores exist in
        #    this methodology, so there is nothing to decouple it from.
        sys_i, usr_i = build_intention_update_prompt(agent, retrieved, timestep)
        int_out = self._chat_json(sys_i, usr_i)

        # 8. optional social post (visible to followers at t+1)
        tweet_text = None
        if social_on:
            system, user = build_tweet_prompt(agent, int_out.get("reasoning", ""), timestep)
            tweet_out = self._chat_json(system, user)
            if tweet_out.get("post") and tweet_out.get("text"):
                tweet_text = tweet_out["text"]

        # ── COMMIT (no LLM calls below) ─────────────────────────────────────
        # Pull the value out of int_out BEFORE mutating agent state, so a
        # KeyError on a malformed LLM response raises before any lesson is
        # attached — keeping a retried agent's re-perception idempotent.
        intention = int_out["fertility_intention"]
        reasoning = int_out.get("reasoning", "")
        for lesson in new_lessons:
            agent.add_lesson(lesson)
        agent.update_intention(intention, timestep, reasoning)
        if tweet_text:
            agent.post_tweet(tweet_text, timestep)

        self._log(f"t={timestep} {agent.agent_id}: "
                  f"intent={agent.fertility_intention_dist}")

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
        """Run one simulation week for every agent. Agents whose week raises
        (after the client's own retries) are retried on their own; if any
        still fail the week is aborted with NO partial save, so --resume
        restarts it cleanly. The whole world state is saved once at the end."""
        self.current_timestep = timestep
        step_start = time.time()
        policy_on, _ = CONDITIONS[self.condition]
        news_items = self.news_schedule.get(timestep, []) if policy_on else []
        if news_items:
            self.logger.debug(f"t={timestep} news: "
                              f"{[n.policy.name for n in news_items]}")

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

        path = self.save()
        self.logger.info(f"Week {timestep} done in {time.time() - step_start:.0f}s, "
                         f"state saved to {path}")

    def run(self, num_timesteps, start_timestep=1):
        """Initialise seed memories (once, if missing), establish the t=0
        baseline, then step `num_timesteps` weeks."""
        if not all(a.seed_lessons for a in self.agents):
            self.initialise_seed_memories()
            self.save()
        self.initialise_baseline()
        self.save()
        for t in range(start_timestep, start_timestep + num_timesteps):
            self._log(f"\n===== Week {t} ({self.condition}, direct/no-TPB) =====")
            self.step(t)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self):
        """Write the full world state to JSON. `methodology` distinguishes
        this from a TPB-mediation run of the same condition at a glance."""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"{self.run_name}.json")
        state = {
            "condition": self.condition,
            "methodology": "direct_no_tpb",
            "current_timestep": self.current_timestep,
            "network": self.network,
            "agents": [a.to_dict() for a in self.agents],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        return path
