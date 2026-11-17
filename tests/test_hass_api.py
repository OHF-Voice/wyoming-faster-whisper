"""Tests for reading names out of Home Assistant.

Runs against a real websocket server on localhost that speaks Home Assistant's
protocol, so the auth handshake, the command/reply pairing, and the shape of the
registry responses are all exercised for real -- only the data is fake.
"""

import json

import pytest

pytest.importorskip("aiohttp")

# pylint: disable=wrong-import-position
from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestServer  # noqa: E402

from wyoming_faster_whisper.hass_api import (  # noqa: E402
    HomeAssistant,
    HomeAssistantError,
)

TOKEN = "good-token"

_EXPOSED = {
    "exposed_entities": {
        "light.office_lamp": {"conversation": True},
        "sensor.ecobee": {"conversation": True},
        "switch.old": {"conversation": True},
        # Not exposed to conversation: not a possible voice target.
        "light.secret": {"conversation": False},
    }
}

_STATES = [
    {"entity_id": "light.office_lamp", "attributes": {"friendly_name": "Office Lamp"}},
    {"entity_id": "sensor.ecobee", "attributes": {"friendly_name": "Ecobee"}},
    {"entity_id": "switch.old", "attributes": {"friendly_name": "Old Switch"}},
    {"entity_id": "light.secret", "attributes": {"friendly_name": "Secret Light"}},
]

_ENTRIES = {
    # No area of its own: it inherits its device's, which is the common case.
    "light.office_lamp": {
        "aliases": ["Desk Lamp"],
        "disabled_by": None,
        "device_id": "dev-lamp",
    },
    "sensor.ecobee": {
        "aliases": ["Thermostat"],
        "disabled_by": None,
        "area_id": "office",
    },
    # Disabled entities are gone from the user's home in every practical sense.
    "switch.old": {
        "aliases": ["Ancient Switch"],
        "disabled_by": "user",
        "area_id": "kitchen",
    },
}

_DEVICES = [{"id": "dev-lamp", "area_id": "office"}]

_AREAS = [
    {
        "area_id": "office",
        "name": "Office",
        "aliases": ["Study"],
        "floor_id": "upstairs",
    },
    # Nothing exposed lives here: its only entity is disabled.
    {"area_id": "kitchen", "name": "Kitchen", "aliases": [], "floor_id": "downstairs"},
]

_FLOORS = [
    {"floor_id": "upstairs", "name": "Upstairs", "aliases": ["Top Floor"]},
    {"floor_id": "downstairs", "name": "Downstairs", "aliases": []},
]

_RESULTS = {
    "homeassistant/expose_entity/list": _EXPOSED,
    "get_states": _STATES,
    "config/entity_registry/get_entries": _ENTRIES,
    "config/device_registry/list": _DEVICES,
    "config/area_registry/list": _AREAS,
    "config/floor_registry/list": _FLOORS,
}


def _app(results=None, fail_command=None, seen=None) -> web.Application:
    """A fake Home Assistant websocket API."""
    results = _RESULTS if results is None else results

    async def handler(request: web.Request) -> web.WebSocketResponse:
        websocket = web.WebSocketResponse()
        await websocket.prepare(request)

        await websocket.send_json({"type": "auth_required", "ha_version": "2026.8.0"})
        msg = json.loads((await websocket.receive()).data)
        if (msg.get("type") != "auth") or (msg.get("access_token") != TOKEN):
            await websocket.send_json({"type": "auth_invalid"})
            return websocket

        await websocket.send_json({"type": "auth_ok"})

        async for raw in websocket:
            msg = json.loads(raw.data)
            msg_type = msg["type"]
            if seen is not None:
                seen.append(msg_type)

            if msg_type == fail_command:
                await websocket.send_json(
                    {
                        "id": msg["id"],
                        "type": "result",
                        "success": False,
                        "error": {"code": "unauthorized", "message": "nope"},
                    }
                )
                continue

            await websocket.send_json(
                {
                    "id": msg["id"],
                    "type": "result",
                    "success": True,
                    "result": results[msg_type],
                }
            )

        return websocket

    app = web.Application()
    app.router.add_get("/api/websocket", handler)
    return app


async def _client(app: web.Application, token: str = TOKEN):
    """Start the fake server and return a client pointed at it, plus the server."""
    server = TestServer(app)
    await server.start_server()
    hass = HomeAssistant(token, api_url=f"http://127.0.0.1:{server.port}/api")
    return hass, server


# --- happy path -----------------------------------------------------------


async def test_names_are_sorted_into_priority_tiers():
    hass, server = await _client(_app())
    try:
        context = await hass.get_context()
    finally:
        await server.close()

    # The office holds both exposed entities -- the lamp through its device, the
    # Ecobee directly -- which also puts the floor it is on in the top tier.
    assert context.used_areas == ["Office", "Study", "Upstairs", "Top Floor"]

    # Each alias rides along with the name it belongs to.
    assert context.priority_entities == ["Office Lamp", "Desk Lamp"]

    # The kitchen's only entity is disabled, and nothing is on its floor.
    assert context.empty_areas == ["Kitchen", "Downstairs"]

    # A sensor is asked about by area far more often than by name.
    assert context.other_entities == ["Ecobee", "Thermostat"]


async def test_an_alias_outranks_a_lower_tier_name():
    """An alias is what the speaker says, so it cannot rank below every name."""
    hass, server = await _client(_app())
    try:
        context = await hass.get_context()
    finally:
        await server.close()

    names = context.all_names()
    assert names.index("Desk Lamp") < names.index("Kitchen")


async def test_an_unnamed_entity_still_contributes_its_alias():
    entries = {
        "light.office_lamp": {"aliases": ["Standing Light"], "disabled_by": None}
    }
    states = [{"entity_id": "light.office_lamp", "attributes": {}}]
    results = {
        **_RESULTS,
        "get_states": states,
        "config/entity_registry/get_entries": entries,
    }

    hass, server = await _client(_app(results=results))
    try:
        context = await hass.get_context()
    finally:
        await server.close()

    assert context.priority_entities == ["Standing Light"]


async def test_the_device_registry_is_only_fetched_when_an_entity_needs_it():
    entries = {"sensor.ecobee": {"disabled_by": None, "area_id": "office"}}
    results = {**_RESULTS, "config/entity_registry/get_entries": entries}

    seen: list = []
    hass, server = await _client(_app(results=results, seen=seen))
    try:
        await hass.get_context()
    finally:
        await server.close()

    assert "config/device_registry/list" not in seen


async def test_unexposed_entities_are_ignored():
    hass, server = await _client(_app())
    try:
        context = await hass.get_context()
    finally:
        await server.close()

    assert "Secret Light" not in context.all_names()


async def test_disabled_entities_and_their_aliases_are_ignored():
    hass, server = await _client(_app())
    try:
        context = await hass.get_context()
    finally:
        await server.close()

    names = context.all_names()
    assert "Old Switch" not in names
    assert "Ancient Switch" not in names


async def test_the_prompt_is_built_from_a_live_fetch():
    hass, server = await _client(_app())
    try:
        context = await hass.get_context()
    finally:
        await server.close()

    prompt = context.whisper_prompt(lambda text: len(text.split()), max_tokens=1000)
    assert prompt == (
        "Office, Study, Upstairs, Top Floor, Office Lamp, Desk Lamp, "
        "Kitchen, Downstairs, Ecobee, Thermostat."
    )


async def test_an_empty_home_yields_no_names():
    empty = {
        "homeassistant/expose_entity/list": {"exposed_entities": {}},
        "get_states": [],
        "config/area_registry/list": [],
        "config/floor_registry/list": [],
    }
    hass, server = await _client(_app(results=empty))
    try:
        context = await hass.get_context()
    finally:
        await server.close()

    assert not context


# --- failures -------------------------------------------------------------


async def test_a_bad_token_raises():
    hass, server = await _client(_app(), token="wrong-token")
    try:
        with pytest.raises(HomeAssistantError):
            await hass.get_context()
    finally:
        await server.close()


async def test_a_refused_command_raises():
    hass, server = await _client(_app(fail_command="get_states"))
    try:
        with pytest.raises(HomeAssistantError):
            await hass.get_context()
    finally:
        await server.close()


async def test_a_non_admin_token_says_so():
    # Home Assistant answers admin-only commands for a non-admin user with
    # code "unauthorized", after auth has already succeeded.
    hass, server = await _client(_app(fail_command="homeassistant/expose_entity/list"))
    try:
        with pytest.raises(HomeAssistantError, match="administrator"):
            await hass.get_context()
    finally:
        await server.close()


async def test_an_unreachable_home_assistant_raises():
    # Nothing is listening on this port.
    hass = HomeAssistant(TOKEN, api_url="http://127.0.0.1:1/api", timeout=2.0)
    with pytest.raises(HomeAssistantError):
        await hass.get_context()


# --- url handling ---------------------------------------------------------


@pytest.mark.parametrize(
    ("api_url", "expected"),
    [
        (
            "http://homeassistant.local:8123/api",
            "ws://homeassistant.local:8123/api/websocket",
        ),
        ("https://ha.example.com/api", "wss://ha.example.com/api/websocket"),
        # A trailing slash must not double up.
        (
            "http://homeassistant.local:8123/api/",
            "ws://homeassistant.local:8123/api/websocket",
        ),
    ],
)
def test_websocket_url_is_derived_from_the_api_url(api_url, expected):
    assert HomeAssistant(TOKEN, api_url=api_url).websocket_api_url == expected


def test_a_non_http_url_is_rejected():
    with pytest.raises(ValueError):
        HomeAssistant(TOKEN, api_url="ftp://homeassistant.local/api")
