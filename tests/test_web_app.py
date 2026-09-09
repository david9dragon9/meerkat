from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from meerkat.events import EventType, ModelResponse, StreamEvent
from meerkat.funnel.spec import FunnelSpec, GateSpec, ResponseSpec
from meerkat.web import (
    BrowserLiveSource,
    LIVE_CAMERA_MEDIA_ID,
    LIVE_SCREEN_MEDIA_ID,
    WEB_BUILD_ID,
    _decode_audio_samples,
    _decode_data_url_frame,
    _plan_payload,
    create_app,
)


def upload(client, name: str = "clip.mp4", data: bytes = b"fake") -> dict:
    """Upload a file the way the browser does, returning the server's descriptor."""
    response = client.post("/api/media", files={"file": (name, data, "application/octet-stream")})
    assert response.status_code == 200, response.text
    return response.json()


def test_upload_describes_the_stored_file(tmp_path) -> None:
    app = create_app(workspace=tmp_path)

    with TestClient(app) as client:
        data = upload(client, "holiday clip.mp4")

    assert data["name"] == "holiday clip.mp4"
    assert data["audio_only"] is False
    assert data["media_url"].startswith(f"/api/media/{data['media_id']}?v=")
    assert data["media_info"]["file_size"] == 4
    assert data["media_info"]["mtime_ns"] > 0


def test_upload_keeps_the_id_opaque_and_never_reuses_the_filename(tmp_path) -> None:
    """Nothing the client sends may become a path component."""
    app = create_app(workspace=tmp_path)

    with TestClient(app) as client:
        data = upload(client, "../../escape.mp4")

    assert "/" not in data["media_id"] and ".." not in data["media_id"]
    assert data["media_id"].endswith(".mp4")


def test_upload_detects_audio_only_media(tmp_path) -> None:
    app = create_app(workspace=tmp_path)

    with TestClient(app) as client:
        assert upload(client, "voices.mp3")["audio_only"] is True


def test_upload_rejects_an_unsupported_file_type(tmp_path) -> None:
    app = create_app(workspace=tmp_path)

    with TestClient(app) as client:
        response = client.post("/api/media", files={"file": ("notes.txt", b"hi", "text/plain")})

    assert response.status_code == 400
    assert ".txt" in response.text


def test_uploaded_media_is_served_back_uncached(tmp_path) -> None:
    app = create_app(workspace=tmp_path)

    with TestClient(app) as client:
        media_id = upload(client)["media_id"]
        response = client.get(f"/api/media/{media_id}")

    assert response.status_code == 200
    assert response.content == b"fake"
    assert response.headers["cache-control"] == "no-store, max-age=0"


def test_unknown_media_id_is_not_found(tmp_path) -> None:
    app = create_app(workspace=tmp_path)

    with TestClient(app) as client:
        assert client.get("/api/media/nope.mp4").status_code == 404


def test_session_media_url_includes_file_version(tmp_path) -> None:
    app = create_app(workspace=tmp_path)

    with TestClient(app) as client:
        media_id = upload(client)["media_id"]
        response = client.post(
            "/api/sessions",
            json={
                "media": media_id,
                "prompt": "let me know when the dog shows up",
                "speak": False,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["media_url"] == f"/api/media/{media_id}?v={data['media_info']['media_version']}"
    assert data["name"] == "clip.mp4"


def test_index_html_substitutes_the_build_marker(tmp_path) -> None:
    app = create_app(workspace=tmp_path)

    with TestClient(app) as client:
        html = client.get("/").text

    assert "__BUILD__" not in html
    assert f"styles.css?v={WEB_BUILD_ID}" in html


def test_web_app_exposes_build_marker(tmp_path) -> None:
    app = create_app(workspace=tmp_path)

    with TestClient(app) as client:
        response = client.get("/api/version")

    assert response.status_code == 200
    assert response.json()["build"].startswith(WEB_BUILD_ID)


def test_browser_live_media_constants_match_the_server() -> None:
    """A live source has nothing to measure, so the client holds these constants.

    They still have to agree with what the server reports when a live session
    starts, or the UI and the run would disagree about the media.
    """
    import json
    import re
    from pathlib import Path

    from meerkat.web import _live_media_info

    script = (Path(_live_media_info.__globals__["__file__"]).parent / "web_static" / "app.js").read_text()
    literal = re.search(r"const LIVE_MEDIA_INFO = (\{.*?\n\});", script, re.S)
    assert literal, "LIVE_MEDIA_INFO not found in app.js"
    from_client = json.loads(re.sub(r"(\w+):", r'"\1":', literal.group(1)))

    assert from_client == _live_media_info()


def test_live_session_ids_are_what_the_browser_offers() -> None:
    from pathlib import Path

    from meerkat.web import _live_media_info

    markup = (Path(_live_media_info.__globals__["__file__"]).parent / "web_static" / "index.html").read_text()

    assert f'value="{LIVE_CAMERA_MEDIA_ID}"' in markup
    assert f'value="{LIVE_SCREEN_MEDIA_ID}"' in markup


def test_web_session_can_be_created_for_live_camera(tmp_path) -> None:
    app = create_app(workspace=tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/sessions",
            json={
                "media": LIVE_CAMERA_MEDIA_ID,
                "prompt": "let me know when I wave",
                "speak": False,
            },
        )
        session = app.state.manager.get(response.json()["session_id"])

    assert response.status_code == 200
    data = response.json()
    assert data["live"] is True
    assert data["media_url"] == ""
    assert data["audio_only"] is False
    assert session.is_live is True
    assert session.media_path is None


def test_web_session_can_be_created_for_live_screen(tmp_path) -> None:
    app = create_app(workspace=tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/sessions",
            json={
                "media": LIVE_SCREEN_MEDIA_ID,
                "prompt": "let me know when the checkout button appears",
                "speak": False,
            },
        )
        session = app.state.manager.get(response.json()["session_id"])

    assert response.status_code == 200
    data = response.json()
    assert data["live"] is True
    assert data["media_url"] == ""
    assert data["audio_only"] is False
    assert session.is_live is True
    assert session.media_path is None


async def test_browser_live_source_emits_browser_frames() -> None:
    playback_event = asyncio.Event()
    playback_event.set()
    source = BrowserLiveSource(playback_event)
    image = np.full((8, 10, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    await source.push_frame(
        "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"),
        stream_time_ms=250,
        sequence_id=3,
    )
    event = await source.events().__anext__()

    assert event is not None
    assert event.type == EventType.VIDEO_FRAME
    assert event.source_id == "browser_live_camera"
    assert event.stream_time_ms == 250
    assert event.sequence_id == 3
    assert event.payload["frame"].shape[:2] == (8, 10)


async def test_browser_live_source_emits_browser_audio() -> None:
    playback_event = asyncio.Event()
    playback_event.set()
    source = BrowserLiveSource(playback_event)
    samples = np.asarray([0, 16384, -16384], dtype="<i2")

    await source.push_audio(
        base64.b64encode(samples.tobytes()).decode("ascii"),
        stream_time_ms=500,
        sequence_id=4,
        sample_rate=16000,
        chunk_ms=500,
    )
    event = await source.events().__anext__()

    assert event.type == EventType.AUDIO_CHUNK
    assert event.source_id == "browser_live_microphone"
    assert event.stream_time_ms == 500
    assert event.sequence_id == 4
    assert event.payload["sample_rate"] == 16000
    assert event.payload["chunk_ms"] == 500
    np.testing.assert_allclose(event.payload["samples"], [0.0, 0.5, -0.5])


def test_decode_data_url_frame_accepts_jpeg_data_url() -> None:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    frame = _decode_data_url_frame(
        "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    )

    assert frame.shape[:2] == (4, 6)


def test_decode_audio_samples_accepts_int16_base64() -> None:
    samples = np.asarray([0, 32767, -32768], dtype="<i2")

    decoded = _decode_audio_samples(base64.b64encode(samples.tobytes()).decode("ascii"))

    np.testing.assert_allclose(decoded, [0.0, 32767 / 32768, -1.0])


def test_web_app_rejects_a_traversing_media_id(tmp_path) -> None:
    (tmp_path.parent / "secret.mp4").write_bytes(b"secret")
    app = create_app(workspace=tmp_path)

    with TestClient(app) as client:
        for media_id in ("../secret.mp4", "..%2Fsecret.mp4", "/etc/passwd"):
            assert client.get(f"/api/media/{media_id}").status_code in {400, 404}


def test_web_session_uses_queue_tuned_defaults(tmp_path) -> None:
    app = create_app(workspace=tmp_path)

    with TestClient(app) as client:
        media_id = upload(client)["media_id"]
        response = client.post(
            "/api/sessions",
            json={
                "media": media_id,
                "prompt": "let me know when the dog shows up",
                "speak": False,
            },
        )
        session = app.state.manager.get(response.json()["session_id"])

    assert session.request.planner_model is None
    assert session.request.verifier_concurrent_requests == 4
    assert session.request.verifier_min_request_interval_ms == 200
    assert session.request.verifier_queued_frame_dedupe_window_ms == 300


def test_web_plan_payload_caps_realtime_transcription_buffer() -> None:
    payload = _plan_payload(
        FunnelSpec(
            goal="audio",
            gates=[
                GateSpec(
                    id="cheap_realtime_transcription",
                    type="local_realtime_transcription",
                    params={
                        "model": "small.en",
                        "buffer_ms": 4000,
                        "sample_interval_ms": 1500,
                    },
                )
            ],
            response=ResponseSpec(on_match_text="ok"),
        )
    )

    assert payload["gates"][0]["params"]["buffer_ms"] == 2000
    assert payload["gates"][0]["params"]["sample_interval_ms"] == 1000


def test_websocket_planning_streams_events_before_playback(tmp_path) -> None:
    app = create_app(workspace=tmp_path)

    with TestClient(app) as client:
        media_id = upload(client)["media_id"]
        response = client.post(
            "/api/sessions",
            json={
                "media": media_id,
                "prompt": "let me know when the dog shows up",
                "no_planner": True,
                "initial_frame_context": False,
                "speak": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "plan" not in data

        with client.websocket_connect(f"/ws/{data['session_id']}") as websocket:
            websocket.send_json({"type": "plan"})
            event_types = []
            for _ in range(12):
                event_type = websocket.receive_json()["type"]
                event_types.append(event_type)
                if event_type == "user_message":
                    break

    assert "planning" in event_types
    assert "log" in event_types
    assert "plan_completed" in event_types
    assert "user_message" in event_types


def test_web_session_subscribes_to_model_responses_before_runner_start(tmp_path) -> None:
    app = create_app(workspace=tmp_path)

    with TestClient(app) as client:
        media_id = upload(client)["media_id"]
        response = client.post(
            "/api/sessions",
            json={
                "media": media_id,
                "prompt": "let me know when the dog shows up",
                "no_planner": True,
                "initial_frame_context": False,
                "speak": False,
            },
        )
        session = app.state.manager.get(response.json()["session_id"])
        client.portal.call(session.start)
        client.portal.call(
            session.runner.bus.publish,
            StreamEvent(
                type=EventType.MODEL_RESPONSE,
                source_id="test",
                stream_time_ms=1234,
                sequence_id=1,
                payload={
                    "model_response": ModelResponse(
                        text="The dog picked up the beach ball.",
                        trigger_gate_id="verify_dog_pickup",
                    )
                },
            ),
        )
        event = client.portal.call(session.queue.get)
        while event.get("trigger_gate_id") != "verify_dog_pickup":
            event = client.portal.call(session.queue.get)

    assert event["message"] == "The dog picked up the beach ball."
    assert event["trigger_gate_id"] == "verify_dog_pickup"


def test_user_messages_carry_response_evidence(tmp_path) -> None:
    """User messages come from typed response events, so they keep their evidence."""
    app = create_app(workspace=tmp_path)

    with TestClient(app) as client:
        media_id = upload(client)["media_id"]
        response = client.post(
            "/api/sessions",
            json={
                "media": media_id,
                "prompt": "let me know when a person appears",
                "no_planner": True,
                "initial_frame_context": False,
                "speak": False,
            },
        )
        session = app.state.manager.get(response.json()["session_id"])
        client.portal.call(session.start)
        client.portal.call(
            session.runner.bus.publish,
            StreamEvent(
                type=EventType.MODEL_RESPONSE,
                source_id="test",
                stream_time_ms=3024,
                sequence_id=1,
                payload={
                    "model_response": ModelResponse(
                        text="A person appeared.",
                        trigger_gate_id="verify_person",
                        evidence={"reason": "verifier returned YES"},
                    )
                },
            ),
        )
        event = client.portal.call(session.queue.get)
        while event.get("type") != "user_message" or event.get("trigger_gate_id") != "verify_person":
            event = client.portal.call(session.queue.get)

    assert event["message"] == "A person appeared."
    assert event["stream_time_ms"] == 3024
    assert event["evidence"] == {"reason": "verifier returned YES"}


def test_frame_evidence_does_not_break_the_event_stream(tmp_path) -> None:
    """Gate evidence carries decoded frames; they must not make an event unsendable.

    A single unserializable payload used to kill the send loop, leaving the UI
    silent for the rest of the session while commands still appeared to work.
    """
    import json

    from meerkat.web import MediaStore, StartRequest, WebSession

    session = WebSession(
        "frames", MediaStore(tmp_path), None, StartRequest(media="clip.mp4", prompt="watch", speak=False)
    )
    session.on_assistant_message(
        "The dog picked up the beach ball.",
        2500,
        "verify",
        {"reason": "matched", "frame": np.zeros((4, 6, 3), dtype=np.uint8), "stream_time_ms": 2500},
    )

    event = session.queue.get_nowait()
    json.dumps(event)  # must not raise
    assert event["evidence"]["frame"] == "<ndarray 4x6x3>"
    assert event["evidence"]["reason"] == "matched"


def test_json_safe_keeps_ordinary_values_intact() -> None:
    from meerkat.web import _json_safe

    payload = {"n": 3, "f": 1.5, "s": "x", "b": True, "none": None, "list": [1, "a"], "nested": {"k": 2}}

    assert _json_safe(payload) == payload


def test_json_safe_unwraps_numpy_scalars() -> None:
    from meerkat.web import _json_safe

    assert _json_safe(np.float32(1.5)) == 1.5
    assert _json_safe(np.int64(7)) == 7


def test_asset_cache_marker_changes_when_an_asset_changes(tmp_path) -> None:
    """A stale marker leaves browsers running the previous UI after an upgrade."""
    from pathlib import Path

    from meerkat.web import WEB_BUILD_ID, _static_build_id

    static_dir = Path(__file__).resolve().parents[1] / "meerkat" / "web_static"
    before = _static_build_id(static_dir)

    assert before.startswith(WEB_BUILD_ID)

    app_js = static_dir / "app.js"
    original = app_js.stat()
    try:
        # Forwards and backwards: either is a change the browser must see.
        for shift in (1_000_000_000, -1_000_000_000):
            os.utime(app_js, ns=(original.st_atime_ns, original.st_mtime_ns + shift))
            assert _static_build_id(static_dir) != before, f"marker unchanged after shift {shift}"
    finally:
        os.utime(app_js, ns=(original.st_atime_ns, original.st_mtime_ns))

    assert _static_build_id(static_dir) == before


def test_index_uses_the_asset_cache_marker(tmp_path) -> None:
    from pathlib import Path

    from meerkat.web import _static_build_id

    static_dir = Path(__file__).resolve().parents[1] / "meerkat" / "web_static"
    app = create_app(workspace=tmp_path)

    with TestClient(app) as client:
        html = client.get("/").text
        reported = client.get("/api/version").json()["build"]

    assert f"app.js?v={_static_build_id(static_dir)}" in html
    assert reported == _static_build_id(static_dir)


def test_video_duration_comes_from_the_container_not_a_frame_estimate(tmp_path) -> None:
    """OpenCV's frame count is an estimate and can be wildly wrong.

    The UI clamps its clock to this value, so an over-long duration lets the
    clock keep running after the picture has stopped.
    """
    from pathlib import Path

    import av

    from meerkat.web import _media_info, _video_track_duration_ms

    clip = Path("examples/media/dog-beach-ball.mp4")
    with av.open(str(clip)) as container:
        stream = next(item for item in container.streams if item.type == "video")
        truth_ms = round(float(stream.duration * stream.time_base) * 1000)

    info = _media_info(clip)

    assert info["video_duration_ms"] == pytest.approx(truth_ms, abs=50)
    assert _video_track_duration_ms(clip) == pytest.approx(truth_ms, abs=50)

    # The frame-count estimate this replaced is off by seconds on this file.
    estimate_ms = round(info["video_frame_count"] / info["video_fps"] * 1000)
    assert abs(estimate_ms - truth_ms) > 1000


def test_duration_tracks_the_picture_not_a_longer_audio_track(tmp_path) -> None:
    """A container often outlives its video track; the clock must follow the picture."""
    import av

    from meerkat.web import _media_info

    source = Path("examples/media/dog-beach-ball.mp4")
    clip = tmp_path / "video-shorter-than-audio.mp4"
    with av.open(str(source)) as inp, av.open(str(clip), "w") as out:
        mapping = {s.index: out.add_stream_from_template(s) for s in inp.streams if s.type in ("video", "audio")}
        for packet in inp.demux([s for s in inp.streams if s.index in mapping]):
            if packet.dts is None:
                continue
            seconds = float(packet.pts * packet.time_base)
            # Keep 3s of picture but 5s of sound.
            if seconds > (3.0 if packet.stream.type == "video" else 5.0):
                if seconds > 5.0:
                    break
                continue
            packet.stream = mapping[packet.stream.index]
            out.mux(packet)

    with av.open(str(clip)) as container:
        container_ms = round(container.duration / 1000)

    duration_ms = _media_info(clip)["video_duration_ms"]

    assert duration_ms < container_ms - 500, "duration followed the audio track, not the picture"


def test_unreadable_media_reports_no_duration(tmp_path) -> None:
    from meerkat.web import _media_info

    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")

    assert _media_info(broken)["video_duration_ms"] is None


def test_favicon_is_served_so_browsers_stop_probing_for_it(tmp_path) -> None:
    app = create_app(workspace=tmp_path)

    with TestClient(app) as client:
        direct = client.get("/favicon.ico")
        declared = client.get("/static/favicon.png")
        html = client.get("/").text

    assert direct.status_code == 200
    assert direct.headers["content-type"] == "image/png"
    assert declared.status_code == 200
    assert 'rel="icon"' in html and "favicon.png" in html
