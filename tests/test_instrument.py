import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
for sub in ("analysis", "sample-mapping", "instrument-builder", "audio-engine"):
    sys.path.insert(0, str(ROOT / sub))

import inventory  # noqa: E402
from audioinv.features import detect_pitch  # noqa: E402
from instrument_builder.build import BuildError, build_instrument, measure_loudness_db  # noqa: E402
from sample_mapping import harvest  # noqa: E402
from sample_mapping.automap import build_pitched_zones  # noqa: E402
from sample_mapping.looping import find_loop  # noqa: E402
from sample_mapping.zones import build_zones  # noqa: E402
from sampler import dsp  # noqa: E402
from sampler.engine import Engine  # noqa: E402
from sampler.instrument import FORMAT, Instrument, InstrumentError, Patch  # noqa: E402
from sampler.midi import MidiError, Note, notes_from_midi, read_midi, write_midi  # noqa: E402

SR = 44100


def harmonic(freq, dur, amp=0.4, decay=0.0, harmonics=(1.0, 0.5, 0.3, 0.15), sr=SR):
    t = np.arange(int(dur * sr)) / sr
    x = sum(a * np.sin(2 * np.pi * freq * (i + 1) * t) for i, a in enumerate(harmonics))
    x = x * np.exp(-t * decay)
    return (amp * x / np.max(np.abs(x))).astype(np.float32)


def midi_hz(m):
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def rms_db(x):
    return 20 * math.log10(max(float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))), 1e-9))


class MidiTests(unittest.TestCase):
    def _roundtrip(self, notes, **kw):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.mid")
            write_midi(path, notes, **kw)
            return notes_from_midi(read_midi(path))

    def test_times_follow_tempo(self):
        out = self._roundtrip([(0, 1, 60, 100), (2, 0.5, 64, 80)], bpm=120)
        self.assertEqual([(n.key, n.vel) for n in out], [(60, 100), (64, 80)])
        self.assertAlmostEqual(out[0].dur, 0.5, places=3)       # one beat at 120 BPM
        self.assertAlmostEqual(out[1].time, 1.0, places=3)      # two beats in
        self.assertAlmostEqual(out[1].dur, 0.25, places=3)

    def test_sustain_pedal_holds_notes(self):
        notes = [(0, 1, 60, 90)]
        held = self._roundtrip(notes, bpm=120, pedal_events=[(0.5, 127), (4, 0)])
        self.assertAlmostEqual(held[0].dur, 2.0, places=2)      # until the pedal comes up at beat 4
        free = self._roundtrip(notes, bpm=120)
        self.assertAlmostEqual(free[0].dur, 0.5, places=3)

    def test_bad_files_raise(self):
        for blob in (b"", b"not midi at all", b"MThd" + b"\x00" * 3,
                     b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\xe7\x28MTrk\x00\x00\x00\x00",   # SMPTE division
                     b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0MTrk\x00\x00\x00\xff"):  # track past end
            with self.assertRaises(MidiError):
                read_midi(blob)


class DspTests(unittest.TestCase):
    def test_cubic_read_follows_a_sine(self):
        x = np.sin(2 * np.pi * np.arange(2000) / 50.0).astype(np.float32)[:, None]
        pos = np.arange(100.3, 160.3, 0.7)
        got = dsp.cubic_read(x, pos)[:, 0]
        want = np.sin(2 * np.pi * pos / 50.0)
        self.assertLess(float(np.max(np.abs(got - want))), 2e-3)

    def test_bake_loop_is_continuous_and_long_enough(self):
        x = harmonic(220.0, 1.5)[:, None]
        found = find_loop(x[:, 0], SR, 220.0)
        self.assertIsNotNone(found)
        baked = dsp.bake_loop(x, found["start"], found["end"], found["xfade"], 5 * SR)
        self.assertGreaterEqual(baked.shape[0], 5 * SR)
        step = np.abs(np.diff(baked[:, 0]))
        self.assertLess(float(step.max()), 1.5 * float(np.percentile(np.abs(np.diff(x[:, 0])), 99.9)))

    def test_one_pole_filters(self):
        t = np.arange(SR) / SR
        low = np.sin(2 * np.pi * 100 * t)[:, None].astype(np.float32)
        high = np.sin(2 * np.pi * 8000 * t)[:, None].astype(np.float32)
        self.assertLess(rms_db(dsp.one_pole(high, 500, SR, "lp")[SR // 2:]), rms_db(high) - 15)
        self.assertGreater(rms_db(dsp.one_pole(low, 500, SR, "lp")[SR // 2:]), rms_db(low) - 3)
        self.assertLess(rms_db(dsp.one_pole(low, 2000, SR, "hp")[SR // 2:]), rms_db(low) - 15)


class LoopTests(unittest.TestCase):
    def test_finds_loop_on_slowly_decaying_tone(self):
        x = harmonic(196.0, 2.5, decay=0.4)
        found = find_loop(x, SR, 196.0)
        self.assertIsNotNone(found)
        self.assertGreater(found["start"], int(0.15 * SR))
        self.assertLess(found["end"], len(x))
        self.assertGreaterEqual(found["score"], 0.85)

    def test_none_for_noise_short_or_unpitched(self):
        rng = np.random.default_rng(3)
        self.assertIsNone(find_loop(rng.standard_normal(2 * SR).astype(np.float32) * 0.1, SR, 220.0))
        self.assertIsNone(find_loop(harmonic(220.0, 0.4), SR, 220.0))
        self.assertIsNone(find_loop(harmonic(220.0, 2.0), SR, None))

    def test_fast_decay_uses_a_shorter_loop_or_none(self):
        x = harmonic(220.0, 2.5, decay=2.5)    # about 22 dB per second
        found = find_loop(x, SR, 220.0)
        if found is not None:
            self.assertLess((found["end"] - found["start"]) / SR, 0.6)


class PatchTests(unittest.TestCase):
    def _zone(self, lk, hk, lv, hv, root, **kw):
        z = {"s": "clean/x/a.wav", "lk": lk, "hk": hk, "lv": lv, "hv": hv, "root": root}
        z.update(kw)
        return z

    def _patch(self, zones, **voice):
        v = {"attack_s": 0.002, "release_s": 0.2, "gain_db": 0.0, "dynamic_range_db": 30.0, "velocity_curve": 1.0}
        v.update(voice)
        return Patch({"id": "p", "voice": v, "zones": zones})

    def test_velocity_layers_and_gap_fallback(self):
        p = self._patch([self._zone(60, 60, 1, 63, 60, s="clean/x/soft.wav"),
                         self._zone(60, 60, 64, 127, 60, s="clean/x/hard.wav")])
        self.assertTrue(p.pick(60, 20)[0]["s"].endswith("soft.wav"))
        self.assertTrue(p.pick(60, 100)[0]["s"].endswith("hard.wav"))
        gap = self._patch([self._zone(60, 60, 1, 40, 60, s="clean/x/soft.wav"),
                           self._zone(60, 60, 80, 127, 60, s="clean/x/hard.wav")])
        self.assertTrue(gap.pick(60, 70)[0]["s"].endswith("hard.wav"))     # 10 away vs 30 away

    def test_round_robin_cycles(self):
        zones = [self._zone(60, 60, 1, 127, 60, s=f"clean/x/rr{i}.wav", rr=[i, 3]) for i in (1, 2, 3)]
        p = self._patch(zones)
        got = [p.pick(60, 90)[0]["s"][-7:-4] for _ in range(7)]
        self.assertEqual(got, ["rr1", "rr2", "rr3", "rr1", "rr2", "rr3", "rr1"])
        p.reset_round_robin()
        self.assertEqual(p.pick(60, 90)[0]["s"][-7:-4], "rr1")

    def test_out_of_range_keys_borrow_nearest_within_limit(self):
        p = self._patch([self._zone(40, 80, 1, 127, 60)], extend_semitones=5)
        self.assertEqual(len(p.pick(84, 90)), 1)
        self.assertEqual(p.pick(86, 90), [])
        self.assertEqual(p.pick(30, 90), [])

    def test_nearest_root_wins_where_zones_overlap(self):
        p = self._patch([self._zone(48, 62, 1, 127, 50, s="clean/x/low.wav"),
                         self._zone(56, 70, 1, 127, 66, s="clean/x/high.wav")])
        self.assertTrue(p.pick(58, 90)[0]["s"].endswith("low.wav"))
        self.assertTrue(p.pick(63, 90)[0]["s"].endswith("high.wav"))

    def test_manifest_validation(self):
        good = {"id": "p", "voice": {}, "zones": [self._zone(60, 60, 1, 127, 60)]}
        with self.assertRaises(InstrumentError):
            Instrument({"format": "something/else", "patches": [good]})
        with self.assertRaises(InstrumentError):
            Instrument({"format": FORMAT, "patches": []})
        for bad_path in ("/etc/passwd", "../outside.wav", "clean/../../x.wav"):
            bad = {"id": "p", "voice": {}, "zones": [self._zone(60, 60, 1, 127, 60, s=bad_path)]}
            with self.assertRaises(InstrumentError):
                Instrument({"format": FORMAT, "patches": [bad]})
        with self.assertRaises(InstrumentError):
            Instrument({"format": FORMAT, "patches": [{"id": "p", "voice": {}, "zones": [
                self._zone(70, 60, 1, 127, 60)]}]})


class SyntheticPackTests(unittest.TestCase):
    """Builds a tiny pack on disk, analyzes it, maps it and plays it back."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.datasets = base / "datasets"
        pack = cls.datasets / "clean" / "testpack"
        (pack / "Samples").mkdir(parents=True)
        # G3 is recorded 30 cents sharp: the declared root says 55 but the audio says 55.3.
        specs = [("C3", 48.0, 0.08), ("C3", 48.0, 0.45), ("G3", 55.3, 0.08), ("G3", 55.3, 0.45)]
        lines = ["<control> default_path=Samples/"]
        for i, (name, pitch, amp) in enumerate(specs):
            layer = "soft" if amp < 0.2 else "loud"
            fname = f"{name}_{layer}.wav"
            sf.write(pack / "Samples" / fname, harmonic(midi_hz(pitch), 2.2, amp=amp), SR, subtype="PCM_24")
            lo, hi, root = (46, 51, 48) if name == "C3" else (52, 58, 55)
            vlo, vhi = (1, 63) if layer == "soft" else (64, 127)
            lines.append(f"<region> sample={fname} lokey={lo} hikey={hi} pitch_keycenter={root} lovel={vlo} hivel={vhi}")
        (pack / "test.sfz").write_text("\n".join(lines) + "\n")
        cls.inv = base / "inventory"
        inventory.process_pack("clean", "testpack", str(pack), str(cls.inv), str(base / "reports"), 1, True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _zones(self, **opts):
        return build_zones(str(self.datasets), "clean", "testpack", "test.sfz", str(self.inv), opts)

    def test_zone_mapping_and_tuning_refinement(self):
        zones, rep = self._zones()
        self.assertEqual(len(zones), 4)
        self.assertEqual(rep["velocity_layers"], 2)
        self.assertEqual(rep["keys"], [46, 58])
        g3 = [z for z in zones if "G3" in z["s"]]
        self.assertTrue(all(abs(z["root"] - 55.3) < 0.04 for z in g3), [z["root"] for z in g3])
        c3 = [z for z in zones if "C3" in z["s"]]
        self.assertTrue(all(abs(z["root"] - 48.0) < 0.04 for z in c3))
        self.assertEqual(rep["declared_vs_detected_pitch"]["within_60_cents"], 4)
        # the sustained, cut-off tones got loops
        self.assertEqual(rep["loops"].get("auto found"), 4, rep["loops"])
        self.assertTrue(all("loop" in z for z in zones))
        # louder layer is the reference (0 dB), the soft one sits below it
        self.assertEqual({z["rl"] for z in zones if z["lv"] == 64}, {0.0})
        self.assertLess(max(z["rl"] for z in zones if z["lv"] == 1), -10.0)

    def test_declared_root_is_kept_when_refinement_is_off(self):
        zones, _ = self._zones(refine_tuning=False)
        self.assertTrue(all(z["root"] == 55 for z in zones if "G3" in z["s"]))

    def _instrument(self, **voice):
        zones, _ = self._zones()
        v = {"attack_s": 0.002, "release_s": 0.25, "gain_db": 0.0, "dynamic_range_db": 30.0, "velocity_curve": 1.0,
             "loop_decay_s": None, **voice}
        manifest = {"format": FORMAT, "name": "t", "patches": [{"id": "p", "voice": v, "zones": zones}]}
        return Engine(Instrument(manifest), str(self.datasets), sr=SR), manifest

    def test_rendered_pitch_is_right_between_and_beyond_zone_roots(self):
        eng, _ = self._instrument()
        errors = []
        for key in range(46, 59):
            audio, _ = eng.render([Note(0.0, 1.0, key, 100)], tail_s=0.2, peak_limit=1e9)
            res = detect_pitch(audio.mean(axis=1), SR, int(0.03 * SR))
            self.assertTrue(res.get("pitched"), key)
            errors.append(abs(res["pitch_midi"] - key) * 100)
        self.assertLess(max(errors), 8.0, errors)

    def test_velocity_is_monotonic_and_follows_the_range(self):
        eng, _ = self._instrument(dynamic_range_db=30.0)
        levels = []
        for vel in range(8, 128, 8):
            audio, _ = eng.render([Note(0.0, 0.8, 50, vel)], tail_s=0.1, peak_limit=1e9)
            levels.append(rms_db(audio.mean(axis=1)[: int(0.4 * SR)]))
        self.assertTrue(all(b > a for a, b in zip(levels, levels[1:])), levels)
        # velocity 8 -> 120 spans about 30 * (1 - 8/127) - 30 * (1 - 120/127) of the curve
        self.assertAlmostEqual(levels[-1] - levels[0], 30 * (112 / 127), delta=3.0)

    def test_held_note_outlasts_the_sample_by_looping(self):
        eng, _ = self._instrument()
        audio, _ = eng.render([Note(0.0, 6.0, 50, 100)], tail_s=0.3, peak_limit=1e9)
        mono = audio.mean(axis=1)
        self.assertFalse(np.isnan(audio).any())
        self.assertGreater(rms_db(mono[int(5.0 * SR):int(5.8 * SR)]), -45.0)   # sample is only 2.2 s long
        eng2, _ = self._instrument()
        short, _ = eng2.render([Note(0.0, 0.5, 50, 100)], tail_s=0.5, peak_limit=1e9)
        tail = short.mean(axis=1)[int(0.5 * SR + 0.3 * SR):]                    # well after release
        self.assertLess(rms_db(tail), rms_db(short.mean(axis=1)[:int(0.4 * SR)]) - 40)

    def test_notes_start_where_asked_and_sum(self):
        eng, _ = self._instrument()
        audio, stats = eng.render([Note(1.0, 0.3, 50, 100), Note(1.0, 0.3, 55, 100)], tail_s=0.2, peak_limit=1e9)
        mono = audio.mean(axis=1)
        self.assertLess(float(np.max(np.abs(mono[: int(0.99 * SR)]))), 1e-6)
        self.assertGreater(float(np.max(np.abs(mono[int(1.0 * SR):int(1.1 * SR)]))), 0.01)
        self.assertEqual(stats["notes_played"], 2)

    def test_peak_limit_scales_instead_of_clipping(self):
        eng, _ = self._instrument(gain_db=20.0)
        audio, stats = eng.render([Note(0.0, 1.0, 50, 127)], tail_s=0.2, peak_limit=0.9)
        self.assertLessEqual(float(np.max(np.abs(audio))), 0.9 + 1e-6)
        self.assertLess(stats["scale"], 1.0)

    def test_loudness_matching_hits_the_target_and_private_tier_is_blocked(self):
        spec = {"name": "t", "target_rms_db": -24.0, "patches": [
            {"id": "a", "source": {"pack": "testpack", "sfz": "test.sfz"}, "voice": {"gain_db": 6.0}},
            {"id": "b", "source": {"pack": "testpack", "sfz": "test.sfz"}, "voice": {"gain_db": -8.0}}]}
        manifest, report = build_instrument(spec, str(self.datasets), str(self.inv))
        self.assertTrue(manifest["distributable"])
        for pid in ("a", "b"):
            self.assertAlmostEqual(measure_loudness_db(manifest, pid, str(self.datasets)), -24.0, delta=0.6)
        blocked = {"name": "t", "patches": [{"id": "a", "source": {"pack": "testpack", "sfz": "test.sfz",
                                                                    "tier": "research-private"}}]}
        with self.assertRaises(BuildError):
            build_instrument(blocked, str(self.datasets), str(self.inv))

    def test_engine_refuses_sample_paths_outside_the_datasets_folder(self):
        eng, _ = self._instrument()
        with self.assertRaises(ValueError):
            eng.load_sample("clean/../../etc/passwd")


def band_db(x, freq, lo_s, hi_s, sr=SR):
    """Level (dB) of a sine-like component near freq in x[lo_s:hi_s]."""
    seg = np.asarray(x[int(lo_s * sr):int(hi_s * sr)], dtype=np.float64)
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) / len(seg)
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    m = (f > freq * 0.97) & (f < freq * 1.03)
    return 20 * math.log10(max(float(spec[m].max()), 1e-9))


class FeatureTests(unittest.TestCase):
    """Mono playing, one-shots, choke groups, and the mapper options that feed them."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.datasets = base / "datasets"
        pack = cls.datasets / "clean" / "featpack"
        (pack / "Samples").mkdir(parents=True)
        t = np.arange(int(2.0 * SR)) / SR
        sf.write(pack / "Samples" / "tone_a.wav", (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32), SR)
        sf.write(pack / "Samples" / "tone_b.wav", (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32), SR)
        sf.write(pack / "Samples" / "hat_open.wav", (0.3 * np.sin(2 * np.pi * 1000.0 * t[:SR])).astype(np.float32), SR)
        sf.write(pack / "Samples" / "hat_closed.wav", (0.3 * np.sin(2 * np.pi * 2000.0 * t[:SR // 5])).astype(np.float32), SR)
        # a bass written an octave above where it sounds: the audio is MIDI 40 (E2), the SFZ says 52
        sf.write(pack / "Samples" / "oct.wav", harmonic(midi_hz(40.0), 2.0, amp=0.4, decay=1.5), SR, subtype="PCM_24")
        (pack / "feat.sfz").write_text(
            "<control> default_path=Samples/ set_cc100=0\n"
            "<region> sample=oct.wav lokey=50 hikey=54 pitch_keycenter=52\n"
            "<region> sample=tone_a.wav key=60 volume=-6\n"
            "<region> sample=hat_open.wav key=46 loop_mode=one_shot group=1 off_by=2\n"
            "<region> sample=hat_closed.wav key=42 loop_mode=one_shot group=2 off_by=1\n"
            "<group> locc100=1\n<region> sample=tone_b.wav key=62\n")
        cls.inv = base / "inventory"
        inventory.process_pack("clean", "featpack", str(pack), str(cls.inv), str(base / "reports"), 1, True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _zones(self, **opts):
        return build_zones(str(self.datasets), "clean", "featpack", "feat.sfz", str(self.inv), opts)

    def _engine(self, zones, **voice):
        v = {"attack_s": 0.002, "release_s": 0.1, "gain_db": 0.0, "dynamic_range_db": 30.0, "velocity_curve": 1.0,
             "loop_decay_s": None, **voice}
        manifest = {"format": FORMAT, "name": "t", "patches": [{"id": "p", "voice": v, "zones": zones}]}
        return Engine(Instrument(manifest), str(self.datasets), sr=SR)

    @staticmethod
    def _zone(sample, key, root, **kw):
        return {"s": f"clean/featpack/Samples/{sample}", "lk": key, "hk": key, "lv": 1, "hv": 127, "root": root, **kw}

    # --- mapper -------------------------------------------------------------------------------
    def test_pack_written_an_octave_off_is_moved_to_its_sounding_pitch(self):
        zones, rep = self._zones(include="oct")
        self.assertEqual(rep["pack_octave"], -1)
        z = zones[0]
        self.assertEqual((z["lk"], z["hk"]), (38, 42))
        self.assertAlmostEqual(z["root"], 40.0, delta=0.06)
        # asking for the pack's own key map keeps the author's octave
        zones, rep = self._zones(include="oct", pitch_convention="pack")
        self.assertEqual((zones[0]["lk"], zones[0]["hk"]), (50, 54))
        self.assertAlmostEqual(zones[0]["root"], 52.0, delta=0.06)

    def test_sounding_key_plays_the_pitch_it_is_named_after(self):
        zones, _ = self._zones(include="oct")
        eng = self._engine(zones)
        for key in (38, 40, 42):
            audio, _ = eng.render([Note(0.0, 0.8, key, 100)], tail_s=0.1, peak_limit=1e9)
            res = detect_pitch(audio.mean(axis=1), SR, int(0.05 * SR))
            self.assertTrue(res.get("pitched"), key)
            self.assertLess(abs(res["pitch_midi"] - key) * 100, 10.0, (key, res))

    def test_unpitched_or_scattered_detections_do_not_move_the_key_map(self):
        zones, rep = self._zones()        # whole pack: tones, hats and the octave-off bass disagree
        self.assertEqual(rep["pack_octave"], 0)
        self.assertEqual(rep["key_shift_semitones"], 0)

    def test_volume_one_shot_and_choke_fields_are_carried_into_zones(self):
        zones, _ = self._zones(pitch_convention="pack", refine_tuning=False)
        by = {z["s"].split("/")[-1]: z for z in zones}
        self.assertEqual(by["tone_a.wav"]["g"], -6.0)
        self.assertEqual((by["hat_open.wav"]["os"], by["hat_open.wav"]["grp"], by["hat_open.wav"]["off_by"]), (1, 1, 2))
        self.assertEqual((by["hat_closed.wav"]["grp"], by["hat_closed.wav"]["off_by"]), (2, 1))
        zones, _ = self._zones(pitch_convention="pack", refine_tuning=False, sfz_volume=False)
        self.assertEqual({z["g"] for z in zones if "tone_a" in z["s"]}, {0.0})

    def test_controller_gated_samples_follow_the_cc_option(self):
        names = lambda **o: {z["s"].split("/")[-1] for z in self._zones(**o)[0]}  # noqa: E731
        self.assertNotIn("tone_b.wav", names())
        self.assertIn("tone_b.wav", names(cc={100: 5}))

    def test_key_range_clips_and_drops_zones(self):
        zones, rep = self._zones(include="oct", key_range=[40, 41])
        self.assertEqual((zones[0]["lk"], zones[0]["hk"]), (40, 41))
        zones, rep = self._zones(include="oct", key_range=[60, 70])
        self.assertEqual(zones, [])
        self.assertEqual(rep["skipped"].get("outside key_range"), 1)

    # --- engine -------------------------------------------------------------------------------
    def test_mono_patch_cuts_the_sounding_note_and_poly_does_not(self):
        zones = [self._zone("tone_a.wav", 57, 57.0), self._zone("tone_b.wav", 69, 69.0)]
        notes = [Note(0.0, 1.6, 57, 100), Note(0.5, 1.0, 69, 100)]
        poly, st_poly = self._engine(zones).render(notes, tail_s=0.2, peak_limit=1e9)
        mono, st_mono = self._engine(zones, mono=True, mono_release_s=0.04).render(notes, tail_s=0.2, peak_limit=1e9)
        held = band_db(poly.mean(axis=1), 220.0, 0.8, 1.4)
        cut = band_db(mono.mean(axis=1), 220.0, 0.8, 1.4)
        self.assertEqual((st_poly["mono_cuts"], st_mono["mono_cuts"]), (0, 1))
        self.assertGreater(held, -45.0)
        self.assertLess(cut, held - 40.0)
        self.assertGreater(band_db(mono.mean(axis=1), 440.0, 0.8, 1.4), -45.0)   # the new note sounds
        self.assertGreater(band_db(mono.mean(axis=1), 220.0, 0.1, 0.45), -45.0)  # and the old one sounded before it

    def test_mono_chord_notes_at_the_same_time_all_sound(self):
        zones = [self._zone("tone_a.wav", 57, 57.0), self._zone("tone_b.wav", 69, 69.0)]
        audio, st = self._engine(zones, mono=True).render([Note(0.0, 1.0, 57, 100), Note(0.0, 1.0, 69, 100)],
                                                           tail_s=0.2, peak_limit=1e9)
        self.assertEqual(st["mono_cuts"], 0)
        self.assertGreater(band_db(audio.mean(axis=1), 220.0, 0.2, 0.8), -45.0)
        self.assertGreater(band_db(audio.mean(axis=1), 440.0, 0.2, 0.8), -45.0)

    def test_one_shot_plays_to_the_end_whatever_the_note_length(self):
        normal = self._zone("tone_b.wav", 69, 69.0)
        shot = self._zone("tone_b.wav", 69, 69.0, os=1)
        short = [Note(0.0, 0.05, 69, 100)]
        a, _ = self._engine([normal]).render(short, tail_s=1.5, peak_limit=1e9)
        b, _ = self._engine([shot]).render(short, tail_s=1.5, peak_limit=1e9)
        self.assertLess(band_db(a.mean(axis=1), 440.0, 0.5, 1.0), -80.0)
        self.assertGreater(band_db(b.mean(axis=1), 440.0, 0.5, 1.0), -45.0)
        self.assertGreater(b.shape[0], int(1.9 * SR))        # the render grew to hold the whole sample

    def test_choke_group_cuts_the_open_hat_when_the_closed_hat_plays(self):
        zones = [self._zone("hat_open.wav", 46, 46.0, os=1, grp=1, off_by=2),
                 self._zone("hat_closed.wav", 42, 42.0, os=1, grp=2, off_by=1)]
        eng = self._engine(zones, choke_s=0.02)
        open_only, _ = eng.render([Note(0.0, 0.1, 46, 100)], tail_s=0.3, peak_limit=1e9)
        both, st = eng.render([Note(0.0, 0.1, 46, 100), Note(0.3, 0.1, 42, 100)], tail_s=0.3, peak_limit=1e9)
        self.assertEqual(st["choked"], 1)
        self.assertGreater(band_db(open_only.mean(axis=1), 1000.0, 0.5, 0.9), -45.0)   # rings on its own
        self.assertLess(band_db(both.mean(axis=1), 1000.0, 0.5, 0.9), -80.0)            # choked at 0.3 s
        self.assertGreater(band_db(both.mean(axis=1), 1000.0, 0.1, 0.28), -45.0)        # sounded until then
        # a note outside the choke relationship does not cut it
        other, st2 = eng.render([Note(0.0, 0.1, 46, 100), Note(0.3, 0.1, 46, 100)], tail_s=0.3, peak_limit=1e9)
        self.assertEqual(st2["choked"], 0)

    def test_key_balance_reduces_level_differences_between_keys(self):
        quiet = self._zone("tone_a.wav", 57, 57.0, g=-12.0)
        loud = self._zone("tone_b.wav", 69, 69.0, g=0.0)
        spec = {"name": "t", "balance": {"keys": [57, 69], "max_db": 12, "strength": 1.0}, "match_loudness": False,
                "patches": []}
        from instrument_builder.build import balance_keys
        patch = {"id": "p", "voice": {"attack_s": 0.002, "release_s": 0.1, "gain_db": 0.0, "dynamic_range_db": 30.0,
                                      "velocity_curve": 1.0, "loop_decay_s": None}, "zones": [quiet, loud]}
        manifest = {"format": FORMAT, "name": "t", "patches": [patch]}
        info = balance_keys(manifest, patch, str(self.datasets), [57, 69], max_db=12.0, strength=1.0)
        self.assertGreater(info["before_spread_db"], 10.0)
        self.assertLess(info["after_spread_db"], 1.5)
        del spec

    def test_stack_patch_plays_the_same_notes_through_transposed_layers(self):
        zones = [self._zone("tone_a.wav", 57, 57.0)]
        zones[0].update({"lk": 40, "hk": 80})
        manifest = {"format": FORMAT, "name": "t", "patches": [
            {"id": "a", "voice": self._voice(), "zones": zones},
            {"id": "s", "voice": self._voice(), "zones": [],
             "stack": [{"patch": "a", "transpose": 0, "gain_db": 0.0}, {"patch": "a", "transpose": 12, "gain_db": -6.0}]}]}
        eng = Engine(Instrument(manifest), str(self.datasets), sr=SR)
        audio, st = eng.render([Note(0.0, 1.0, 57, 100)], patch="s", tail_s=0.2, peak_limit=1e9)
        self.assertEqual(st["layers"], 2)
        low, high = band_db(audio.mean(axis=1), 220.0, 0.2, 0.8), band_db(audio.mean(axis=1), 440.0, 0.2, 0.8)
        self.assertGreater(low, -45.0)
        self.assertGreater(high, -45.0)
        self.assertAlmostEqual(low - high, 6.0, delta=1.0)    # the octave layer is 6 dB down

    def test_bad_stacks_are_rejected(self):
        base = {"id": "a", "voice": self._voice(), "zones": [self._zone("tone_a.wav", 57, 57.0)]}
        def load(*extra):
            return Instrument({"format": FORMAT, "name": "t", "patches": [base, *extra]})
        with self.assertRaises(InstrumentError):    # unknown layer
            load({"id": "s", "voice": self._voice(), "zones": [], "stack": [{"patch": "nope"}]})
        with self.assertRaises(InstrumentError):    # a stack inside a stack
            load({"id": "s", "voice": self._voice(), "zones": [], "stack": [{"patch": "a"}]},
                 {"id": "t", "voice": self._voice(), "zones": [], "stack": [{"patch": "s"}]})
        with self.assertRaises(InstrumentError):    # zones and layers together
            load({"id": "s", "voice": self._voice(), "zones": base["zones"], "stack": [{"patch": "a"}]})
        with self.assertRaises(InstrumentError):    # absurd transpose
            load({"id": "s", "voice": self._voice(), "zones": [], "stack": [{"patch": "a", "transpose": 99}]})

    def test_decay_option_shortens_notes(self):
        zones = [self._zone("tone_a.wav", 57, 57.0)]
        plain, _ = self._engine(zones).render([Note(0.0, 1.5, 57, 100)], tail_s=0.2, peak_limit=1e9)
        fast, _ = self._engine(zones, decay_s=0.1).render([Note(0.0, 1.5, 57, 100)], tail_s=0.2, peak_limit=1e9)
        drop = band_db(plain.mean(axis=1), 220.0, 0.9, 1.1) - band_db(fast.mean(axis=1), 220.0, 0.9, 1.1)
        self.assertGreater(drop, 30.0)      # e^(-1/0.1 s) at 1 s is about -87 dB; the ratio is large either way
        # the decay already acts inside the first 0.1 s: mean power of e^(-2t/0.1) over 0.02-0.1 s is -4.8 dB
        early = band_db(plain.mean(axis=1), 220.0, 0.02, 0.1) - band_db(fast.mean(axis=1), 220.0, 0.02, 0.1)
        self.assertAlmostEqual(early, 4.8, delta=1.0)

    @staticmethod
    def _voice(**kw):
        return {"attack_s": 0.002, "release_s": 0.1, "gain_db": 0.0, "dynamic_range_db": 30.0, "velocity_curve": 1.0,
                "loop_decay_s": None, **kw}


class AutomapTests(unittest.TestCase):
    """Mapping a pack that has no SFZ, from its file names."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.datasets = base / "datasets"
        pack = cls.datasets / "clean" / "autopack" / "Horn" / "sus"
        pack.mkdir(parents=True)
        # names sit an octave below the sounding pitch (A2 sounds 220 Hz = MIDI 57), like the VSCO brass
        t = np.arange(int(3.0 * SR)) / SR
        for name, midi in (("A2", 57), ("C3", 60), ("E3", 64)):
            for vel, amp in ((1, 0.1), (2, 0.4)):
                for rr in (1, 2):
                    tone = amp * np.sin(2 * np.pi * midi_hz(midi) * t) * (1 + 0.02 * rr)
                    sf.write(pack / f"Horn_sus_{name}_v{vel}_rr{rr}.wav", tone.astype(np.float32), SR)
        sf.write(pack / "Horn_sus_noise.wav", (0.1 * np.random.default_rng(1).standard_normal(SR)).astype(np.float32), SR)
        cls.inv = base / "inventory"
        inventory.process_pack("clean", "autopack", str(cls.datasets / "clean" / "autopack"), str(cls.inv),
                               str(base / "reports"), 1, True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _zones(self, **opts):
        return build_pitched_zones(str(self.datasets), "clean", "autopack", str(self.inv), {"include": "Horn/sus/", **opts})

    def test_keys_follow_the_sounding_pitch_not_the_octave_in_the_names(self):
        zones, rep = self._zones()
        self.assertEqual(rep["pack_octave"], 1)
        roots = sorted({round(z["root"]) for z in zones})
        self.assertEqual(roots, [57, 60, 64])
        for z in zones:
            self.assertLessEqual(z["lk"], round(z["root"]))
            self.assertGreaterEqual(z["hk"], round(z["root"]))

    def test_note_ranges_meet_halfway_and_extend_past_the_ends(self):
        zones, rep = self._zones(key_span=3)
        spans = {round(z["root"]): (z["lk"], z["hk"]) for z in zones}
        self.assertEqual(spans[57], (54, 58))        # three keys below, then halfway to 60
        self.assertEqual(spans[60], (59, 62))
        self.assertEqual(spans[64], (63, 67))

    def test_velocity_ids_become_equal_bands_and_repeats_become_round_robins(self):
        zones, rep = self._zones()
        self.assertEqual(rep["velocity_layers"], 2)
        self.assertEqual({(z["lv"], z["hv"]) for z in zones}, {(1, 64), (65, 127)})
        self.assertEqual(rep["round_robin_groups"], 6)
        self.assertEqual(sorted(z["rr"] for z in zones if round(z["root"]) == 60 and z["lv"] == 1), [[1, 2], [2, 2]])

    def test_unmatched_names_are_skipped_and_reported(self):
        zones, rep = self._zones()
        self.assertEqual(rep["skipped"], {"name did not match": 1})
        self.assertEqual(len(zones), 12)

    def test_loop_and_one_shot_options(self):
        zones, rep = self._zones(loop="stable")
        self.assertEqual(rep["loops"].get("found"), 12)
        for z in zones:
            start, end, xf = z["loop"]
            self.assertTrue(0.0 < start < end <= 3.0)
        zones, rep = self._zones(one_shot=True, loop="stable")
        self.assertTrue(all(z.get("os") == 1 and "loop" not in z for z in zones))

    def test_a_pattern_that_matches_nothing_is_an_error(self):
        with self.assertRaises(ValueError):
            self._zones(include="Nothing/")
        with self.assertRaises(ValueError):
            build_pitched_zones(str(self.datasets), "clean", "autopack", str(self.inv), {})

    def test_mapped_notes_play_at_their_own_pitch(self):
        zones, _ = self._zones()
        manifest = {"format": FORMAT, "name": "t", "patches": [{"id": "p", "voice": FeatureTests._voice(), "zones": zones}]}
        eng = Engine(Instrument(manifest), str(self.datasets), sr=SR)
        for key in (57, 59, 60, 64, 66):
            audio, _ = eng.render([Note(0.0, 0.8, key, 100)], tail_s=0.1, peak_limit=1e9)
            res = detect_pitch(audio.mean(axis=1), SR, int(0.05 * SR))
            self.assertLess(abs(res["pitch_midi"] - key) * 100, 8.0, (key, res))


class AutomapFillVelocityTests(unittest.TestCase):
    """Sparse packs (harvested notes) have keys with only some of the velocity layers."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.datasets = base / "datasets"
        pack = cls.datasets / "clean" / "sparse" / "g"
        pack.mkdir(parents=True)
        # keys: A3 has layers 1 and 3, C4 only layer 3, E4 only layer 1, G4 all three layers
        have = {57: (1, 3), 60: (3,), 64: (1,), 67: (1, 2, 3)}
        names = {57: "A3", 60: "C4", 64: "E4", 67: "G4"}
        for midi, layers in have.items():
            for v in layers:
                sf.write(pack / f"Gtr_{names[midi]}_v{v}.wav", harmonic(midi_hz(midi), 0.5, 0.2 * v, 6.0), SR)
        cls.inv = base / "inventory"
        inventory.process_pack("clean", "sparse", str(cls.datasets / "clean" / "sparse"), str(cls.inv),
                               str(base / "reports"), 1, True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _bands(self, **opts):
        zones, _ = build_pitched_zones(str(self.datasets), "clean", "sparse", str(self.inv), {"include": "g/", **opts})
        out = {}
        for z in zones:
            out.setdefault(round(z["root"]), []).append((z["lv"], z["hv"]))
        return {k: sorted(v) for k, v in out.items()}

    def test_without_filling_a_key_keeps_only_its_own_bands(self):
        bands = self._bands()
        self.assertEqual(bands[60], [(86, 127)])      # a lone loud layer leaves soft playing silent

    def test_filling_lets_each_key_cover_the_whole_velocity_range(self):
        bands = self._bands(fill_velocity=True)
        self.assertEqual(bands[60], [(1, 127)])
        self.assertEqual(bands[64], [(1, 127)])
        self.assertEqual(bands[57], [(1, 64), (65, 127)])        # two layers meet in the middle of the gap
        self.assertEqual(bands[67], [(1, 42), (43, 85), (86, 127)])    # a complete key is left as it was


class HarvestTests(unittest.TestCase):
    """Cutting single notes out of a loop stem."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        stem = np.zeros(int(14 * SR), dtype=np.float32)

        def put(t, x):
            a = int(t * SR)
            stem[a:a + len(x)] += x[:len(stem) - a]

        def pluck(midi, amp, dur=0.6, cents=0.0):
            x = harmonic(midi_hz(midi + cents / 100.0), dur, amp, 6.0)
            x[:int(0.002 * SR)] *= np.linspace(0, 1, int(0.002 * SR))
            return x

        # single notes: A3 three times at different strengths, D4, E4, a B3 that is 30 cents sharp
        for i, (midi, amp, cents) in enumerate([(57, 0.5, 0), (62, 0.5, 0), (64, 0.4, 0), (57, 0.12, 0),
                                                (59, 0.45, 30), (57, 0.5, 0), (62, 0.1, 0)]):
            put(0.5 + 0.8 * i, pluck(midi, amp, cents=cents))
        # a strummed chord (C3 E3 G3, 12 ms between strings) and a burst of noise: neither is a single pitch
        for j, m in enumerate((48, 52, 55)):
            put(7.0 + 0.012 * j, pluck(m, 0.4, 0.9))
        put(9.0, (0.3 * np.random.default_rng(3).standard_normal(int(0.3 * SR)) * np.exp(-np.arange(int(0.3 * SR)) / 3000)).astype(np.float32))
        cls.stem_path = base / "Dry" / "stem.wav"
        cls.stem_path.parent.mkdir()
        sf.write(cls.stem_path, stem, SR)
        cls.out = base / "out"
        cls.report = harvest.harvest([str(cls.stem_path)], str(cls.out), "t", {"layers": 2})
        cls.files = sorted(os.listdir(cls.out))
        cls.wavs = [f for f in cls.files if f.endswith(".wav")]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_finds_the_single_notes_and_names_them_for_the_automapper(self):
        import re
        self.assertGreaterEqual(len(self.wavs), 6)
        names = set()
        for f in self.wavs:
            m = re.fullmatch(r"t_([A-G]#?-?\d+)_v(\d+)_rr(\d+)\.wav", f)
            self.assertIsNotNone(m, f)
            names.add(m.group(1))
        self.assertTrue({"A3", "D4", "E4", "B3"} <= names, names)

    def test_chords_and_noise_are_left_out(self):
        names = " ".join(self.wavs)
        for chord_note in ("C3", "E3", "G3"):
            self.assertNotIn(f"_{chord_note}_", names)
        self.assertGreater(sum(self.report["rejected"].values()), 0)

    def test_every_note_is_on_pitch_even_when_the_loop_was_sharp(self):
        for f in self.wavs:
            x, sr = sf.read(self.out / f, always_2d=True)
            note = f.split("_")[1]
            midi = 12 * (int(note[-1]) + 1) + {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7,
                                                "G#": 8, "A": 9, "A#": 10, "B": 11}[note[:-1]]
            res = detect_pitch(x.mean(axis=1), sr, int(0.002 * sr))
            self.assertTrue(res["pitched"], f)
            self.assertAlmostEqual(res["pitch_midi"], midi, delta=0.12, msg=f)    # within 12 cents, so B3 (30 cents sharp) was corrected

    def test_notes_start_and_end_without_clicks(self):
        for f in self.wavs:
            x, _ = sf.read(self.out / f, always_2d=True)
            peak = float(np.max(np.abs(x)))
            self.assertLess(float(np.max(np.abs(x[:4]))), 0.05 * peak, f)
            self.assertLess(float(np.max(np.abs(x[-4:]))), 0.002, f)

    def test_different_strengths_of_one_pitch_become_layers_and_repeats_become_round_robins(self):
        a3 = [f for f in self.wavs if "_A3_" in f]
        self.assertGreaterEqual(len(a3), 3)
        self.assertTrue(any("_v1_" in f for f in a3) and any("_v2_" in f for f in a3), a3)
        rr = {f.split("_")[-1] for f in a3 if "_v2_" in f}
        self.assertGreaterEqual(len(rr), 2, a3)       # two loud A3 notes: two round robins of the loud layer

    def test_round_robins_of_a_layer_play_at_one_level(self):
        loud = sorted(f for f in self.wavs if "_A3_v2_" in f)
        levels = [rms_db(sf.read(self.out / f)[0][int(0.01 * SR):int(0.21 * SR)]) for f in loud]
        self.assertLess(max(levels) - min(levels), 1.0, levels)

    def test_the_report_lists_what_was_kept(self):
        with open(self.out / "harvest-report.json", encoding="utf-8") as fh:
            rep = json.load(fh)
        self.assertEqual(rep["files_written"], len(self.wavs))
        self.assertEqual(len(rep["notes"]), len(self.wavs))
        self.assertEqual(rep["family"], "t")


@unittest.skipUnless((ROOT / "datasets" / "clean" / "salamander-grand").is_dir()
                     and (ROOT / "generated-instruments" / "afrobeat" / "piano" / "instrument.json").exists(),
                     "sample packs are not downloaded here (see scripts/fetch_datasets.sh)")
class AfrobeatPianoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = Instrument.load(str(ROOT / "generated-instruments" / "afrobeat" / "piano" / "instrument.json"))
        cls.eng = Engine(cls.inst, str(ROOT / "datasets"), sr=SR)

    def test_every_patch_plays_in_tune(self):
        for pid, patch in self.inst.patches.items():
            keys = sorted(patch.by_key)
            for key in (keys[len(keys) // 4], keys[len(keys) // 2], keys[3 * len(keys) // 4]):
                audio, _ = self.eng.render([Note(0.0, 1.0, key, 100)], patch=pid, tail_s=0.2, peak_limit=1e9)
                res = detect_pitch(audio.mean(axis=1), SR, int(0.05 * SR))
                self.assertTrue(res.get("pitched"), (pid, key))
                self.assertLess(abs(res["pitch_midi"] - key) * 100, 12.0, (pid, key, res))

    def test_velocity_is_monotonic_on_every_patch(self):
        for pid in self.inst.patches:
            levels = []
            for vel in range(10, 128, 12):
                audio, _ = self.eng.render([Note(0.0, 0.8, 60, vel)], patch=pid, tail_s=0.1, peak_limit=1e9)
                levels.append(rms_db(audio.mean(axis=1)[: int(0.4 * SR)]))
            self.assertTrue(all(b > a for a, b in zip(levels, levels[1:])), (pid, levels))

    def test_credits_name_both_ccby_packs(self):
        packs = {c["pack"]: c for c in self.inst.manifest["credits"]}
        self.assertEqual(set(packs), {"salamander-grand", "greg-sullivan-epianos"})
        self.assertTrue(all("CC-BY" in c["license"] for c in packs.values()))
        self.assertTrue(self.inst.manifest["distributable"])


@unittest.skipUnless((ROOT / "datasets" / "clean" / "pastabass").is_dir()
                     and (ROOT / "generated-instruments" / "afrobeat" / "bass" / "instrument.json").exists(),
                     "sample packs are not downloaded here (see scripts/fetch_datasets.sh)")
class AfrobeatBassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = Instrument.load(str(ROOT / "generated-instruments" / "afrobeat" / "bass" / "instrument.json"))
        cls.eng = Engine(cls.inst, str(ROOT / "datasets"), sr=SR)

    @staticmethod
    def _autocorr(x, lag):
        a, b = x[:-lag], x[lag:]
        return float(np.dot(a, b) / math.sqrt(float(np.dot(a, a) * np.dot(b, b)) + 1e-12))

    def test_electric_patches_play_in_tune_and_in_the_right_octave(self):
        for pid in ("finger", "pick", "muted", "round", "hollow", "short-scale"):
            for key in (36, 43, 48):
                audio, _ = self.eng.render([Note(0.0, 1.0, key, 100)], patch=pid, tail_s=0.2, peak_limit=1e9)
                mono = audio.mean(axis=1)
                res = detect_pitch(mono, SR, int(0.1 * SR))
                self.assertTrue(res.get("pitched"), (pid, key))
                self.assertLess(abs(res["pitch_midi"] - key) * 100, 12.0, (pid, key, res))
                # an independent look: the waveform repeats at the period of the key
                seg = mono[int(0.1 * SR):int(0.6 * SR)]
                lag = SR / midi_hz(key)
                best = max(self._autocorr(seg, int(round(lag * f))) for f in (0.99, 1.0, 1.01))
                self.assertGreater(best, 0.8, (pid, key, best))

    def test_every_patch_is_monophonic(self):
        for pid in self.inst.patches:
            _, st = self.eng.render([Note(0.0, 1.0, 36, 100), Note(0.4, 0.5, 43, 100)], patch=pid, tail_s=0.1)
            self.assertEqual(st["mono_cuts"], 1, pid)

    def test_velocity_is_monotonic_on_every_patch(self):
        for pid in self.inst.patches:
            levels = []
            for vel in range(10, 128, 12):
                audio, _ = self.eng.render([Note(0.0, 0.8, 40, vel)], patch=pid, tail_s=0.1, peak_limit=1e9)
                levels.append(rms_db(audio.mean(axis=1)[: int(0.4 * SR)]))
            self.assertTrue(all(b > a for a, b in zip(levels, levels[1:])), (pid, levels))

    def test_levels_are_balanced_across_the_keyboard(self):
        for pid in self.inst.patches:
            levels = []
            for key in range(30, 51, 4):
                audio, _ = self.eng.render([Note(0.0, 1.0, key, 100)], patch=pid, tail_s=0.1, peak_limit=1e9)
                levels.append(rms_db(audio.mean(axis=1)[int(0.02 * SR):int(0.52 * SR)]))
            self.assertLess(max(levels) - min(levels), 7.0, (pid, [round(v, 1) for v in levels]))

    def test_credits_and_licences(self):
        packs = {c["pack"] for c in self.inst.manifest["credits"]}
        self.assertEqual(packs, {"pastabass", "black-and-blue-basses", "sneakybass"})
        self.assertTrue(self.inst.manifest["distributable"])


@unittest.skipUnless((ROOT / "datasets" / "clean" / "shinyguitar").is_dir()
                     and (ROOT / "generated-instruments" / "afrobeat" / "guitar" / "instrument.json").exists(),
                     "sample packs are not downloaded here (see scripts/fetch_datasets.sh)")
class AfrobeatGuitarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = Instrument.load(str(ROOT / "generated-instruments" / "afrobeat" / "guitar" / "instrument.json"))
        cls.eng = Engine(cls.inst, str(ROOT / "datasets"), sr=SR)

    def test_every_patch_plays_in_tune(self):
        # the guitar-banjo was recorded about 40 cents sharp and must be corrected; its plucks are hard for a
        # pitch detector, so it is checked on the keys where the detector locks on
        keys = {"ganjo": (46, 52, 58, 64, 67)}
        for pid in self.inst.patches:
            for key in keys.get(pid, (45, 52, 60, 67)):
                audio, _ = self.eng.render([Note(0.0, 1.0, key, 100)], patch=pid, tail_s=0.2, peak_limit=1e9)
                res = detect_pitch(audio.mean(axis=1), SR, int(0.05 * SR))
                self.assertTrue(res.get("pitched"), (pid, key))
                self.assertLess(abs(res["pitch_midi"] - key) * 100, 10.0, (pid, key, res))

    def test_velocity_is_monotonic_on_every_patch(self):
        for pid in self.inst.patches:
            levels = []
            for vel in range(10, 128, 12):
                audio, _ = self.eng.render([Note(0.0, 0.8, 52, vel)], patch=pid, tail_s=0.1, peak_limit=1e9)
                levels.append(rms_db(audio.mean(axis=1)[: int(0.4 * SR)]))
            self.assertTrue(all(b > a for a, b in zip(levels, levels[1:])), (pid, levels))

    def test_pickup_and_microphone_patches_use_different_recordings(self):
        def folders(pid):
            return {z["s"].split("/")[-2] for z in self.inst.manifest["patches"][[p["id"] for p in self.inst.manifest["patches"]].index(pid)]["zones"]}
        self.assertEqual(folders("clean"), {"electric"})
        self.assertEqual(folders("mic"), {"acoustic"})

    def test_chop_starts_like_the_clean_patch_and_dies_away_much_faster(self):
        levels = {}
        for pid in ("clean", "chop"):
            audio, _ = self.eng.render([Note(0.0, 2.0, 57, 100)], patch=pid, tail_s=0.3, peak_limit=1e9)
            m = audio.mean(axis=1)
            levels[pid] = (rms_db(m[int(0.05 * SR):int(0.15 * SR)]), rms_db(m[int(0.8 * SR):int(0.9 * SR)]))
        self.assertLess(abs(levels["clean"][0] - levels["chop"][0]), 3.0)
        self.assertLess(levels["chop"][1], levels["clean"][1] - 25.0)

    def test_levels_are_balanced_across_the_keyboard(self):
        for pid in self.inst.patches:
            levels = []
            for key in range(43, 77, 4):
                if key not in self.inst.patches[pid].by_key:
                    continue
                audio, _ = self.eng.render([Note(0.0, 1.0, key, 100)], patch=pid, tail_s=0.1, peak_limit=1e9)
                levels.append(rms_db(audio.mean(axis=1)[int(0.02 * SR):int(0.52 * SR)]))
            self.assertLess(max(levels) - min(levels), 7.0, (pid, [round(v, 1) for v in levels]))

    def test_credits(self):
        self.assertEqual({c["pack"] for c in self.inst.manifest["credits"]}, {"shinyguitar", "ganjo"})
        self.assertTrue(self.inst.manifest["distributable"])


@unittest.skipUnless((ROOT / "datasets" / "clean" / "vsco2ce").is_dir()
                     and (ROOT / "generated-instruments" / "afrobeat" / "horns" / "instrument.json").exists(),
                     "sample packs are not downloaded here (see scripts/fetch_datasets.sh)")
class AfrobeatHornsTests(unittest.TestCase):
    RANGES = {"trumpet-stab": (60, 78), "trumpet-sustain": (60, 78), "trombone-stab": (40, 62),
              "trombone-sustain": (40, 62), "sax": (55, 77)}

    @classmethod
    def setUpClass(cls):
        cls.inst = Instrument.load(str(ROOT / "generated-instruments" / "afrobeat" / "horns" / "instrument.json"))
        cls.eng = Engine(cls.inst, str(ROOT / "datasets"), sr=SR)

    def test_notes_play_in_tune(self):
        for pid, (lo, hi) in self.RANGES.items():
            checked = 0
            for key in range(lo, hi + 1, 3):
                audio, _ = self.eng.render([Note(0.0, 1.0, key, 100)], patch=pid, tail_s=0.3, peak_limit=1e9)
                res = detect_pitch(audio.mean(axis=1), SR, int(0.15 * SR))
                if res.get("pitched"):         # stabs are short and the detector sometimes declines
                    checked += 1
                    self.assertLess(abs(res["pitch_midi"] - key) * 100, 20.0, (pid, key, res))
            self.assertGreaterEqual(checked, 5, pid)

    def test_sustained_patches_hold_a_long_note_without_dropping_out_or_pulsing(self):
        for pid in ("trumpet-sustain", "trombone-sustain", "sax"):
            lo, hi = self.RANGES[pid]
            for key in range(lo, hi + 1, 4):
                audio, _ = self.eng.render([Note(0.0, 6.0, key, 100)], patch=pid, tail_s=0.3, peak_limit=1e9)
                m = audio.mean(axis=1)
                frames = [rms_db(m[int(t * SR):int((t + 0.1) * SR)]) for t in np.arange(1.0, 5.8, 0.1)]
                self.assertGreater(min(frames), -70.0, (pid, key, "sound ended early"))
                self.assertLess(max(frames) - min(frames), 5.0, (pid, key, "level pulses"))

    def test_velocity_is_monotonic_on_every_sampled_patch(self):
        for pid, (lo, hi) in self.RANGES.items():
            levels = []
            for vel in range(10, 128, 12):
                audio, _ = self.eng.render([Note(0.0, 0.8, (lo + hi) // 2, vel)], patch=pid, tail_s=0.1, peak_limit=1e9)
                levels.append(rms_db(audio.mean(axis=1)[: int(0.5 * SR)]))
            # a velocity-layer switch may leave a tiny step (trumpet-stab: 0.12 dB); a real dip is audible
            self.assertTrue(all(b > a - 0.5 for a, b in zip(levels, levels[1:])), (pid, levels))
            self.assertGreater(levels[-1] - levels[0], 12.0, (pid, levels))

    def test_stabs_play_to_the_end_but_sustained_notes_follow_the_key(self):
        short = [Note(0.0, 0.05, 66, 100)]
        stab, _ = self.eng.render(short, patch="trumpet-stab", tail_s=0.6, peak_limit=1e9)
        held, _ = self.eng.render(short, patch="trumpet-sustain", tail_s=0.6, peak_limit=1e9)
        self.assertGreater(rms_db(stab.mean(axis=1)[int(0.25 * SR):int(0.45 * SR)]), -60.0)
        self.assertLess(rms_db(held.mean(axis=1)[int(0.6 * SR):int(0.8 * SR)]), -80.0)

    def test_horn_section_is_trumpet_sax_and_an_octave_down_trombone(self):
        section = self.inst.patches["section"]
        self.assertEqual([(l["patch"], l["transpose"]) for l in section.stack],
                         [("trumpet-stab", 0), ("sax", 0), ("trombone-stab", -12)])
        audio, st = self.eng.render([Note(0.0, 1.0, 69, 100)], patch="section", tail_s=0.5, peak_limit=1e9)
        self.assertEqual(st["layers"], 3)
        m = audio.mean(axis=1)
        self.assertGreater(band_db(m, midi_hz(69), 0.1, 0.5), band_db(m, 3000.0, 0.1, 0.5))     # the played note is there
        self.assertGreater(band_db(m, midi_hz(57), 0.05, 0.6), -70.0)                           # and the trombone's octave below

    def test_loudness_is_matched_across_patches(self):
        levels = {}
        for pid, (lo, hi) in self.RANGES.items():
            audio, _ = self.eng.render([Note(0.0, 0.8, (lo + hi) // 2, 100)], patch=pid, tail_s=0.5, peak_limit=1e9)
            levels[pid] = rms_db(audio.mean(axis=1)[: int(0.8 * SR)])
        self.assertLess(max(levels.values()) - min(levels.values()), 8.0, levels)

    def test_credits(self):
        self.assertEqual({c["pack"] for c in self.inst.manifest["credits"]}, {"vsco2ce", "weresax"})
        self.assertTrue(self.inst.manifest["distributable"])


if __name__ == "__main__":
    unittest.main()
