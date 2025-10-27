"""Actor system primitives for the Totem runtime."""

from __future__ import annotations

from collections import deque
from typing import Callable, Iterable, Optional

from ..constants import EFFECT_GRADES
from .core import Effect


class OwnedMessage:
    """A message that must be moved exactly once across actors."""

    def __init__(self, payload, capability, message_id):
        self.payload = payload
        self.capability = capability
        self.message_id = message_id
        self._moved = False

    def move_payload(self):
        if self._moved:
            raise RuntimeError(f"Message {self.message_id} has already been moved")
        self._moved = True
        return self.payload

    def __repr__(self):  # pragma: no cover - representation helper
        target = getattr(self.capability, "actor_id", "?")
        return f"<OwnedMessage id={self.message_id}→{target} moved={self._moved}>"


class ActorCapability:
    """Capability used to send messages to a specific actor."""

    def __init__(self, actor_system: "ActorSystem", actor_id: str):
        self.actor_system = actor_system
        self.actor_id = actor_id

    def send(self, message: OwnedMessage):
        return self.actor_system.send(self, message)

    def __repr__(self):  # pragma: no cover - representation helper
        return f"<Capability {self.actor_id}>"


class Actor:
    """Single actor with a mailbox and effect-local log."""

    def __init__(self, actor_id: str, behavior: Callable[[OwnedMessage], Effect]):
        self.actor_id = actor_id
        self.behavior = behavior
        self.mailbox: deque = deque()
        self.local_log: list[str] = []
        self.local_grade_index = EFFECT_GRADES.index("pure")

    def enqueue(self, payload):
        self.mailbox.append(payload)

    def drain(self):
        delivered = 0
        logs: list[str] = []
        grade_index = self.local_grade_index
        while self.mailbox:
            payload = self.mailbox.popleft()
            effect = self.behavior(payload)
            grade_index = max(grade_index, EFFECT_GRADES.index(effect.grade))
            logs.extend(effect.log)
            delivered += 1
        self.local_grade_index = grade_index
        self.local_log.extend(logs)
        return delivered, logs, grade_index


def default_actor_behavior(payload) -> Effect:
    return Effect("state", {"last_message": payload}, [f"echo:{payload}"])


class ActorSystem:
    """Ownership-safe actor system with move-only message passing."""

    def __init__(self):
        self.actors: dict[str, Actor] = {}
        self._actor_counter = 0
        self._message_counter = 0
        self._public_log: list[str] = []

    def spawn(
        self, behavior: Optional[Callable[[OwnedMessage], Effect]] = None
    ) -> ActorCapability:
        behavior = behavior or default_actor_behavior
        actor_id = f"actor_{self._actor_counter}"
        self._actor_counter += 1
        actor = Actor(actor_id, behavior)
        self.actors[actor_id] = actor
        return ActorCapability(self, actor_id)

    def next_message_id(self) -> int:
        mid = self._message_counter
        self._message_counter += 1
        return mid

    def send(self, capability: ActorCapability, message: OwnedMessage) -> Effect:
        if message.capability is not capability:
            raise RuntimeError("Message capability does not match the target actor")
        payload = message.move_payload()
        actor = self.actors.get(capability.actor_id)
        if actor is None:
            raise RuntimeError(f"Unknown actor {capability.actor_id}")
        actor.enqueue(payload)
        log_entry = f"send:{capability.actor_id}:msg{message.message_id}"
        return Effect("sys", True, [log_entry])

    def run_until_idle(self) -> Effect:
        delivered = 0
        logs: list[str] = []
        highest_grade = EFFECT_GRADES.index("pure")

        while True:
            iteration_delivered = 0
            iteration_logs: list[str] = []
            for actor_id, actor in self.actors.items():
                count, local_logs, grade_idx = actor.drain()
                if not count and not local_logs:
                    continue
                iteration_delivered += count
                highest_grade = max(highest_grade, grade_idx)
                iteration_logs.extend(f"{actor_id}:{entry}" for entry in local_logs)

            if not iteration_delivered and not iteration_logs:
                break

            delivered += iteration_delivered
            logs.extend(iteration_logs)

        prefix = f"run:delivered={delivered}"
        combined = [prefix] + logs if logs else [prefix]
        self._public_log = logs
        return Effect("sys", self, combined)

    @property
    def last_public_log(self) -> Iterable[str]:
        return list(self._public_log)


__all__ = [
    "Actor",
    "ActorCapability",
    "ActorSystem",
    "OwnedMessage",
    "default_actor_behavior",
]
