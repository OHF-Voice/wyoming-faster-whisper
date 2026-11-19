"""Read the names in a Home Assistant install over the websocket API.

Read-only and recognition-only: this fetches the names a speaker can actually
say -- exposed entities, the areas and floors they live in -- and never calls a
service or handles an intent. The result is a
[RecognitionContext](vocabulary.py) that biases speech-to-text.

Only *conversation-exposed* entities are collected. Every entity in the house
would put hundreds of names the speaker cannot mean into a prompt with room for
a few dozen, crowding out the ones they can.

Even so, an exposed home can overrun the prompt budget on its own, so the names
are also sorted into the priority tiers of a
[RecognitionContext](vocabulary.py): which area holds something exposed, and
which domain an entity belongs to, are both known only here.

Requires the ``hass`` extra (aiohttp).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, Iterable, List, Optional, Set
from urllib.parse import urlparse, urlunparse

import aiohttp

from .const import HASS_API_URL
from .vocabulary import RecognitionContext, clean_names

_LOGGER = logging.getLogger(__name__)

Command = Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]]

# Domains whose entities get named out loud, and whose names are therefore the
# ones worth spending prompt budget on. Everything else falls to a lower tier --
# in practice that is mostly ``sensor``/``binary_sensor``, which a large home
# exposes by the hundred and which are usually asked about by area ("what's the
# temperature in the office") rather than by their own name.
#
# Membership is the whole heuristic; order within the set is irrelevant. Moving
# a domain in or out of here is the one knob for tuning what survives
# truncation.
PRIORITY_DOMAINS = frozenset(
    {
        "light",
        "switch",
        "fan",
        "media_player",
        "climate",
        "scene",
        "todo",
    }
)


@dataclass
class _Entity:
    """One exposed entity, reduced to what the tiers are decided from."""

    entity_id: str
    name: str = ""
    aliases: List[str] = field(default_factory=list)
    area_id: Optional[str] = None

    @property
    def domain(self) -> str:
        return self.entity_id.split(".", 1)[0]

    @property
    def is_priority(self) -> bool:
        return self.domain in PRIORITY_DOMAINS


class HomeAssistantError(Exception):
    """Home Assistant refused a request or the connection failed."""


class HomeAssistant:
    """Read-only client for the names speech-to-text biases toward."""

    def __init__(
        self,
        token: str,
        api_url: str = HASS_API_URL,
        timeout: float = 10.0,
    ) -> None:
        self.token = token
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

        parsed = urlparse(self.api_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

        scheme = "wss" if parsed.scheme == "https" else "ws"
        self.websocket_api_url = urlunparse(
            parsed._replace(
                scheme=scheme,
                path=f"{parsed.path}/websocket",
                params="",
                query="",
                fragment="",
            )
        )

    async def get_context(self) -> RecognitionContext:
        """Fetch the current names. Raises HomeAssistantError on failure."""
        current_id = 0

        def next_id() -> int:
            nonlocal current_id
            current_id += 1
            return current_id

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.ws_connect(
                    self.websocket_api_url, max_msg_size=0
                ) as websocket:

                    async def command(payload: Dict[str, Any]) -> Dict[str, Any]:
                        """One request, one reply. We subscribe to nothing, so
                        nothing else can arrive in between."""
                        await websocket.send_json({"id": next_id(), **payload})
                        msg = await websocket.receive_json()
                        if not msg.get("success"):
                            error = msg.get("error") or {}
                            if error.get("code") == "unauthorized":
                                # The token authenticated, so it is valid; the
                                # command itself is admin-only (expose_entity/list
                                # is @require_admin in Home Assistant).
                                raise HomeAssistantError(
                                    f"{payload['type']} was refused as unauthorized. "
                                    "The access token must belong to a Home "
                                    "Assistant administrator"
                                )

                            raise HomeAssistantError(f"{payload['type']} failed: {msg}")

                        return msg

                    await self._authenticate(websocket)
                    return await self._load(command)
        except HomeAssistantError:
            raise
        except Exception as exc:
            raise HomeAssistantError(f"Failed to load names: {exc}") from exc

    async def _authenticate(self, websocket: Any) -> None:
        msg = await websocket.receive_json()
        if msg.get("type") != "auth_required":
            raise HomeAssistantError(f"Expected auth_required, got {msg}")

        await websocket.send_json({"type": "auth", "access_token": self.token})
        msg = await websocket.receive_json()
        if msg.get("type") != "auth_ok":
            raise HomeAssistantError(f"Authentication failed: {msg}")

    async def _load(self, command: Command) -> RecognitionContext:
        # Entities the user exposed to conversation. Nothing else is a possible
        # target of a voice command.
        msg = await command({"type": "homeassistant/expose_entity/list"})
        exposed: Set[str] = {
            entity_id
            for entity_id, info in (msg["result"]["exposed_entities"] or {}).items()
            if info.get("conversation")
        }

        # The displayed (friendly) name is what Home Assistant itself matches and
        # what a user reading their dashboard will say. The registry's
        # name/original_name may have had the device prefix stripped, leaving
        # "Blinds" where the entity shows as "Bedroom Blinds"; biasing toward
        # that fragment would put a bare cover-class word into the prompt.
        friendly: Dict[str, str] = {}
        msg = await command({"type": "get_states"})
        for state in msg["result"]:
            entity_id = state["entity_id"]
            if entity_id not in exposed:
                continue

            attributes = state.get("attributes") or {}
            name = " ".join((attributes.get("friendly_name") or "").split())
            if name:
                friendly[entity_id] = name

        # Aliases only come with the extended registry entries, so exposed
        # entities are fetched a second time to get them. An alias is what a
        # speaker actually says when the integration's own name is not it.
        entries: Dict[str, Dict[str, Any]] = {}
        if exposed:
            msg = await command(
                {
                    "type": "config/entity_registry/get_entries",
                    "entity_ids": sorted(exposed),
                }
            )
            entries = {k: v for k, v in msg["result"].items() if v}

        # An entity's area is usually its device's; only an explicit override
        # lives on the entity itself. The device registry is the whole house, so
        # it is only fetched when an exposed entity actually needs it.
        device_areas: Dict[str, str] = {}
        if any(
            entry.get("device_id") and not entry.get("area_id")
            for entry in entries.values()
        ):
            msg = await command({"type": "config/device_registry/list"})
            device_areas = {
                device["id"]: device["area_id"]
                for device in msg["result"]
                if device.get("id") and device.get("area_id")
            }

        entities: List[_Entity] = []
        unnamed: List[str] = []
        for entity_id in sorted(exposed):
            entry = entries.get(entity_id) or {}
            if entry.get("disabled_by") is not None:
                # Disabled entities are gone from the home in every practical
                # sense, including as evidence that their area is in use.
                continue

            name = friendly.get(entity_id) or entry.get("name") or ""
            if not name:
                name = entry.get("original_name") or ""

            if not name:
                # Still counts toward its area being in use: "turn off the
                # office" reaches it even though it has no name to bias toward.
                unnamed.append(entity_id)

            entities.append(
                _Entity(
                    entity_id=entity_id,
                    name=name,
                    aliases=list(entry.get("aliases") or []),
                    area_id=entry.get("area_id")
                    or device_areas.get(entry.get("device_id") or ""),
                )
            )

        msg = await command({"type": "config/area_registry/list"})
        areas = list(msg["result"])

        msg = await command({"type": "config/floor_registry/list"})
        floors = list(msg["result"])

        # An area earns the top tier by holding something the speaker can
        # actually command; a floor earns it through such an area.
        used_area_ids = {entity.area_id for entity in entities if entity.area_id}
        used_floor_ids = {
            area.get("floor_id")
            for area in areas
            if area.get("floor_id") and (area.get("area_id") in used_area_ids)
        }

        context = RecognitionContext(
            used_areas=clean_names(
                _place_names(areas, "area_id", used_area_ids, in_use=True),
                _place_names(floors, "floor_id", used_floor_ids, in_use=True),
            ),
            priority_entities=clean_names(
                _entity_names(entity for entity in entities if entity.is_priority)
            ),
            empty_areas=clean_names(
                _place_names(areas, "area_id", used_area_ids, in_use=False),
                _place_names(floors, "floor_id", used_floor_ids, in_use=False),
            ),
            other_entities=clean_names(
                _entity_names(entity for entity in entities if not entity.is_priority)
            ),
        )
        # One line per fetch: this runs once per utterance, so anything per-entity
        # here would bury the rest of the log. Aliases are counted separately
        # even though they are folded into the tiers, because "the alias I added
        # is not in the prompt" is the question this log has to answer.
        _LOGGER.debug(
            "Loaded names from Home Assistant: %s areas in use, %s priority "
            "entity names, %s empty areas, %s other entity names "
            "(%s aliases included)%s",
            len(context.used_areas),
            len(context.priority_entities),
            len(context.empty_areas),
            len(context.other_entities),
            sum(len(entity.aliases) for entity in entities),
            (
                f" (skipped {len(unnamed)} unnamed: {', '.join(unnamed)})"
                if unnamed
                else ""
            ),
        )
        return context


def _entity_names(entities: Iterable[_Entity]) -> List[str]:
    """Each entity's name followed by its aliases.

    Keeping them adjacent means the budget cuts whole entities: it can never
    keep "Floor Lamp" while dropping the "standing light" the speaker actually
    says.
    """
    names: List[str] = []
    for entity in entities:
        if entity.name:
            names.append(entity.name)

        # An unnamed entity can still have an alias, and that alias is then the
        # only way to say it.
        names.extend(entity.aliases)

    return names


def _place_names(
    records: Iterable[Dict[str, Any]],
    id_key: str,
    used_ids: Set[Any],
    in_use: bool,
) -> List[str]:
    """Names and aliases of the areas (or floors) on one side of ``used_ids``.

    Registry order is kept, so the prompt is stable across fetches.
    """
    names: List[str] = []
    for record in records:
        if (record.get(id_key) in used_ids) != in_use:
            continue

        names.append(record.get("name") or "")
        names.extend(record.get("aliases") or [])

    return names


__all__ = [
    "HASS_API_URL",
    "PRIORITY_DOMAINS",
    "HomeAssistant",
    "HomeAssistantError",
]
