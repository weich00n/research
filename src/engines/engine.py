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
import time

from sandbox.lesson import Lesson
from utils.logging_utils import setup_logger
from sandbox.prompts import (
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
                 news_schedule=None, output_dir="outputs", run_name=None,
                 verbose=True):
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {list(CONDITIONS)}")
        self.agents = agents
        self.agents_by_id = {a.agent_id: a for a in agents}
        self.network = network  # {agent_id: [followed agent_ids]}
        self.condition = condition
        self.llm = llm
        self.scorer = scorer  # LLM_judge.RelevanceScorer
        self.news_schedule = news_schedule or {}
        self.output_dir = output_dir
        self.run_name = run_name or f"run_{condition}"
        self.verbose = verbose
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
            for mem in out["memories"][:5]:
                lesson = Lesson(
                    agent_id=agent.agent_id,
                    memory_text=mem["memory_text"],
                    created_timestep=0,
                    source_type="profile_seed",
                    importance=mem.get("importance", 0.5),
                    memory_class="seed",
                    relevance=self.scorer.score(mem["memory_text"]),
                )
                agent.add_lesson(lesson)
            self._log(f"{agent.agent_id}: {len(agent.seed_lessons)} seed memories")

    # ── One timestep ───────────────────────────────────────────────────────

    def _perceive(self, agent, message_text, message_kind, source_type,
                  timestep, source_agent_id=None):
        """Steps 3+5: turn an incoming message into a relevance-scored memory."""
        system, user = build_perception_prompt(agent, message_text, message_kind)
        out = self.llm.chat_json(system, user)
        lesson = Lesson(
            agent_id=agent.agent_id,
            memory_text=out["memory_text"],
            created_timestep=timestep,
            source_type=source_type,
            importance=out.get("importance", 0.5),
            memory_class="simulation",
            source_agent_id=source_agent_id,
            relevance=self.scorer.score(out["memory_text"]),
        )
        agent.add_lesson(lesson)
        return lesson

    def step(self, timestep):
        """Run one simulation week for every agent (CLAUDE.md steps 1-9).

        Per agent: perceive policy news (1) and friends' t-1 posts (2) into
        memories, retrieve the most relevant ones (4-6), ask the LLM for the new
        TPB scores + reflection (7a) and — in a separate call that never sees
        those scores — the new fertility-intention distribution (7b), and
        optionally post a tweet (8). The whole world state is saved once at the
        end of the week (9).
        """
        self.current_timestep = timestep
        step_start = time.time()
        policy_on, social_on = CONDITIONS[self.condition]
        news_items = self.news_schedule.get(timestep, []) if policy_on else []
        if news_items:
            self.logger.debug(f"t={timestep} news: "
                              f"{[n.policy.name for n in news_items]}")

        for agent in self.agents:
            new_messages = []

            # 1. policy news
            for news in news_items:
                self._perceive(agent, news.text, "news report", "policy_news", timestep)
                new_messages.append(f"[policy news] {news.text}")

            # 2. social posts from followed agents, posted at t-1
            if social_on:
                for followed_id in self.network.get(agent.agent_id, []):
                    followed = self.agents_by_id[followed_id]
                    for tweet in followed.tweets:
                        if tweet.created_timestep == timestep - 1:
                            self._perceive(agent, tweet.text, "social media post",
                                           "social_post", timestep,
                                           source_agent_id=followed_id)
                            new_messages.append(f"[post by a friend] {tweet.text}")

            # 4-6. retrieval
            retrieved, construct_map = agent.retrieve(timestep)
            self.logger.debug(f"t={timestep} {agent.agent_id}: read "
                              f"{len(new_messages)} messages, retrieved "
                              f"{len(retrieved)} memories {construct_map}")

            # 7a. TPB update (attitude/norm/pbc + reflection) — intention NOT
            #     generated here.
            sys_t, usr_t = build_tpb_update_prompt(agent, retrieved, new_messages, timestep)
            tpb_out = self.llm.chat_json(sys_t, usr_t)
            # 7b. Fertility intention — generated independently, WITHOUT seeing the
            #     numeric TPB scores, so the TPB->intention link is measured
            #     (mediation), not instructed.
            sys_i, usr_i = build_intention_update_prompt(agent, retrieved, new_messages, timestep)
            int_out = self.llm.chat_json(sys_i, usr_i)
            agent.update_belief_state(
                tpb_out["attitude_score"],
                tpb_out["subjective_norm_score"],
                tpb_out["pbc_score"],
                int_out["fertility_intention"],
                timestep,
            )
            for lesson in retrieved:
                lesson.used_in_update = True

            reflection_text = tpb_out.get("reflection_memory") or ""
            if reflection_text:
                agent.add_lesson(Lesson(
                    agent_id=agent.agent_id,
                    memory_text=reflection_text,
                    created_timestep=timestep,
                    source_type="reflection",
                    importance=REFLECTION_IMPORTANCE,
                    memory_class="simulation",
                    relevance=self.scorer.score(reflection_text),
                ))

            # 8. optional social post (visible to followers at t+1)
            if social_on:
                system, user = build_tweet_prompt(agent, reflection_text, timestep)
                tweet_out = self.llm.chat_json(system, user)
                if tweet_out.get("post") and tweet_out.get("text"):
                    agent.post_tweet(tweet_out["text"], timestep)

            self._log(f"t={timestep} {agent.agent_id}: "
                      f"att={agent.belief_state['attitude_score']:.1f} "
                      f"norm={agent.belief_state['subjective_norm_score']:.1f} "
                      f"pbc={agent.belief_state['pbc_score']:.1f}")

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
