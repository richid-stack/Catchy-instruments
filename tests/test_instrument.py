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


if __name__ == "__main__":
    unittest.main()
