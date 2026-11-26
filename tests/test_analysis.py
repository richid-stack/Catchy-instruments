import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import inventory  # noqa: E402
from audioinv import features  # noqa: E402
from audioinv.features import analyze_file, yin  # noqa: E402
from audioinv.nameparse import midi_to_name, parse_name  # noqa: E402
from audioinv.link import AudioIndex, layer_stats  # noqa: E402
from audioinv.sfzparse import key_to_midi, parse_sfz, read_bank_defines  # noqa: E402

SR = 44100


def tone(freq, dur=1.5, harmonics=(1.0, 0.5, 0.25), decay=None, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    x = sum(a * np.sin(2 * np.pi * freq * (i + 1) * t) for i, a in enumerate(harmonics))
    if decay:
        x = x * np.exp(-t * decay)
    return (0.5 * x / np.max(np.abs(x))).astype(np.float32)


class NameParseTests(unittest.TestCase):
    def test_note_velocity_round_robin(self):
        h = parse_name("Steinway/Steinway_C#4_v2_rr1.wav")
        self.assertEqual(h["name_midi"], 61)
        self.assertEqual(h["name_note"], "C#4")
        self.assertEqual(h["name_vel"], 2)
        self.assertEqual(h["name_rr"], 1)

    def test_flats_and_extremes(self):
        self.assertEqual(parse_name("Piano_Bb3.flac")["name_midi"], 58)
        self.assertEqual(parse_name("Piano_A0.wav")["name_midi"], 21)
        self.assertEqual(parse_name("Pad_C-1.wav")["name_midi"], 0)

    def test_dynamics_and_articulations(self):
        h = parse_name("Cello/Sustain/Cello_G3_f_vibrato.flac")
        self.assertEqual(h["name_midi"], 55)
        self.assertEqual(h["name_dyn"], "f")
        self.assertIn("sustain", h["artic"])
        self.assertIn("vibrato", h["artic"])

    def test_folder_fallback_and_false_positives(self):
        h = parse_name("Strat/E2/hit.wav")
        self.assertEqual(h["name_midi"], 40)
        self.assertEqual(h["name_note_src"], "folder")
        self.assertNotIn("name_midi", parse_name("01.BD.808.wav"))
        self.assertNotIn("name_midi", parse_name("SD2_snare.wav"))

    def test_midi_to_name(self):
        self.assertEqual(midi_to_name(60), "C4")
        self.assertEqual(midi_to_name(21), "A0")

    def test_note_only_names(self):
        self.assertEqual(parse_name("Wurlitzer EP200/Samples/db4mp.flac")["name_midi"], 61)
        self.assertEqual(parse_name("Samples/a1pp.flac")["name_midi"], 33)
        self.assertEqual(parse_name("Samples/c2ff.flac")["name_midi"], 36)
        self.assertEqual(parse_name("Samples/057_A3.flac")["name_midi"], 57)
        for word in ("bd1.wav", "808.wav", "Bass3_hit.wav", "Cello_pizz.wav"):
            self.assertNotIn("name_midi", parse_name("Kit/" + word), word)


class InventoryTests(unittest.TestCase):
    def test_release_samples_trust_the_name(self):
        rec = {"pitch_midi": 69.0, "name_midi": 49, "artic": ["release"]}
        inventory._reconcile(rec)
        self.assertEqual(rec["note_midi"], 49)
        self.assertEqual(rec["note_source"], "name")
        self.assertNotIn("name_agrees", rec)

    def test_detected_pitch_wins_and_octave_shift_is_reported(self):
        rec = {"pitch_midi": 46.43, "name_midi": 58}
        inventory._reconcile(rec)
        self.assertEqual(rec["note_name"], "A#2")
        self.assertEqual(rec["note_source"], "detected")
        self.assertFalse(rec["name_agrees"])
        self.assertEqual(rec["name_octave_shift"], -1)

    def test_sfz_root_summary(self):
        def rec(diff):
            return {"path": "x.wav", "frames": 44100, "sr": 44100, "format": "WAV", "subtype": "PCM_16",
                    "channels": 1, "env": "decaying", "sfz_root_diff": diff}
        g = inventory.summarize_group([rec(0.1), rec(-0.3), rec(12.05), rec(3.0)])
        self.assertEqual(g["sfz_root_vs_pitch"],
                         {"compared": 4, "within_60_cents": 2, "octave_off": 1, "other": 1,
                          "by_octave": {"0": 2, "1": 1}})
        g = inventory.summarize_group([rec(-11.95), rec(-12.1), rec(-12.0)])
        self.assertEqual(g["sfz_root_vs_pitch"]["by_octave"], {"-1": 3})


class SfzTests(unittest.TestCase):
    def test_key_names(self):
        self.assertEqual(key_to_midi("c4"), 60)
        self.assertEqual(key_to_midi("db3"), 49)
        self.assertEqual(key_to_midi("57"), 57)
        self.assertIsNone(key_to_midi("zzz"))

    def test_inheritance_defines_and_default_path(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "a.sfz").write_text(
                "<control> default_path=samples/\n"
                "#define $LOW 36\n"
                "<global> ampeg_release=0.3\n"
                "<group> lovel=1 hivel=64\n"
                "<region> sample=A 3.wav key=57\n"
                "<group> lovel=65 hivel=127\n"
                "<region> sample=A3_hard.wav lokey=$LOW hikey=60 pitch_keycenter=c4 seq_position=2 seq_length=2\n"
            )
            regions, _ = parse_sfz(str(root / "a.sfz"), str(root))
        self.assertEqual(len(regions), 2)
        first, second = regions
        self.assertEqual(first["sample"], "samples/A 3.wav")
        self.assertEqual((first["lokey"], first["hikey"], first["root"]), (57, 57, 57))
        self.assertEqual((first["lovel"], first["hivel"]), (1, 64))
        self.assertEqual((second["lokey"], second["hikey"], second["root"]), (36, 60, 60))
        self.assertEqual((second["lovel"], second["hivel"]), (65, 127))
        self.assertEqual((second["seq_position"], second["seq_length"]), (2, 2))

    def test_include_and_escape_guards(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "pack"
            root.mkdir()
            (Path(d) / "outside.sfz").write_text("<region> sample=x.wav key=60\n")
            (root / "common.sfz").write_text("<region> sample=c.wav key=61\n")
            (root / "main.sfz").write_text(
                '#include "common.sfz"\n'
                '#include "../outside.sfz"\n'
                "<region> sample=../../etc/passwd key=62\n"
                "<region> sample=ok.wav key=63\n"
            )
            regions, warnings = parse_sfz(str(root / "main.sfz"), str(root))
        self.assertEqual(sorted(r["sample"] for r in regions), ["c.wav", "ok.wav"])
        self.assertTrue(warnings)

    def test_midline_includes_and_defines_apply_in_order(self):
        # Salamander style: a group line pulls in a velocity file that defines $VEL, then a
        # region template that uses it. Defines are replaced where they are used, not before.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "Data").mkdir()
            (root / "Data" / "vel_1.txt").write_text("#define $VEL v1\n")
            (root / "Data" / "vel_2.txt").write_text("#define $VEL v2\n")
            (root / "Data" / "region.txt").write_text("<region> lokey=60 hikey=60 sample=C4$VEL.$EXT\n")
            (root / "main.sfz").write_text(
                "#define $EXT flac\n"
                "<control> default_path=Samples/\n"
                '<group> #include "Data/vel_1.txt" lovel=1 hivel=50 #include "Data/region.txt"\n'
                '<group> #include "Data/vel_2.txt" lovel=51 hivel=127 #include "Data/region.txt"\n'
            )
            regions, warnings = parse_sfz(str(root / "main.sfz"), str(root))
        self.assertEqual([r["sample"] for r in regions], ["Samples/C4v1.flac", "Samples/C4v2.flac"])
        self.assertEqual([(r["lovel"], r["hivel"]) for r in regions], [(1, 50), (51, 127)])

    def test_longest_define_wins_and_names_are_whole_words(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "a.sfz").write_text(
                "#define $VEL v1\n#define $VELTRACK 99\n"
                "<region> sample=x$VEL.wav key=60 amp_veltrack=$VELTRACK tune=$VELX\n")
            regions, _ = parse_sfz(str(root / "a.sfz"), str(root))
        self.assertEqual(regions[0]["sample"], "xv1.wav")

    def test_host_defines_and_case_insensitive_include(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "Programs").mkdir()
            (root / "Programs" / "Part.sfz").write_text("<region> sample=a.wav key=60\n")
            (root / "Programs" / "main.sfz").write_text(
                '<control> default_path=$sample_dir/\n#include "part.sfz"\n')
            (root / "Pack.bank.xml").write_text('<Bank><Define name="$sample_dir" value="../Samples"/></Bank>')
            defines = read_bank_defines(str(root))
            included = set()
            regions, _ = parse_sfz(str(root / "Programs" / "main.sfz"), str(root), defines=defines,
                                   included=included)
        self.assertEqual(defines, {"$sample_dir": "../Samples"})
        self.assertEqual(regions[0]["sample"], "Samples/a.wav")
        self.assertEqual(included, {"Programs/Part.sfz"})

    def test_include_bomb_is_capped(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "loop.sfz").write_text('#include "loop.sfz"\n#include "loop.sfz"\n<region> sample=a.wav key=60\n')
            regions, warnings = parse_sfz(str(root / "loop.sfz"), str(root))
        self.assertLess(len(regions), 10_000)
        self.assertTrue(warnings)


class LinkTests(unittest.TestCase):
    def test_resolve_exact_extension_and_suffix(self):
        idx = AudioIndex(["Samples/acoustic/Db2_vl1.wav", "UC/A/Piano_A0.flac", "UC/B/Piano_A0.flac",
                          "Other/x/one.flac"])
        self.assertEqual(idx.resolve("samples/ACOUSTIC/db2_vl1.wav"), "Samples/acoustic/Db2_vl1.wav")
        self.assertEqual(idx.resolve("Samples/acoustic/db2_vl1.flac"), "Samples/acoustic/Db2_vl1.wav")
        self.assertEqual(idx.fuzzy_links, 0)
        # Folder variable was not filled in: match on the trailing components.
        self.assertEqual(idx.resolve("Programs/$dir/acoustic\\db2_vl1.wav"), "Samples/acoustic/Db2_vl1.wav")
        self.assertEqual(idx.fuzzy_links, 1)
        # Same file name in two folders is ambiguous; a longer tail disambiguates.
        self.assertIsNone(idx.resolve("Piano_A0.flac"))
        self.assertEqual(idx.resolve("wrong/A/Piano_A0.flac"), "UC/A/Piano_A0.flac")
        self.assertIsNone(idx.resolve("nothing/here.wav"))
        self.assertIsNone(idx.resolve("Piano_A0.flac", fuzzy=False))

    def _r(self, key, lv, hv, **kw):
        base = {"sfz": "a.sfz", "sample": "s.wav", "lokey": key, "hikey": key, "lovel": lv, "hivel": hv,
                "lorand": 0.0, "hirand": 1.0, "seq_position": 1, "seq_length": 1}
        base.update(kw)
        return base

    def test_layer_stats_counts_layers_and_round_robins(self):
        regions = [
            self._r(60, 0, 63, lorand=0.0, hirand=0.5), self._r(60, 0, 63, lorand=0.5, hirand=1.0),
            self._r(60, 64, 127, seq_position=1, seq_length=3), self._r(60, 64, 127, seq_position=2, seq_length=3),
            self._r(60, 64, 127, seq_position=3, seq_length=3),
            self._r(60, 0, 127, trigger="release"),   # release sample: ignored
            self._r(60, 0, 127),                      # full-range add-on next to split layers: ignored
        ]
        stats = layer_stats(regions)
        self.assertEqual(stats["velocity_layers_max"], 2)
        self.assertEqual(stats["round_robins_max"], 3)
        self.assertEqual(stats["keys_with_regions"], 1)

    def test_layer_stats_single_layer_and_empty(self):
        self.assertEqual(layer_stats([self._r(40, 0, 127)])["velocity_layers_max"], 1)
        self.assertEqual(layer_stats([self._r(40, 0, 127, trigger="release")]), {})


class AudioTests(unittest.TestCase):
    def test_yin_on_sine(self):
        seg = tone(440.0, dur=0.3, harmonics=(1.0,))[:8192]
        f0, conf = yin(seg, SR)
        self.assertAlmostEqual(f0, 440.0, delta=1.0)
        self.assertGreater(conf, 0.95)

    def _write(self, d, name, data, channels=1):
        path = str(Path(d) / name)
        if channels == 2:
            data = np.stack([data, data], axis=1)
        sf.write(path, data, SR, subtype="PCM_24")
        return path

    def test_pitched_decaying_stereo(self):
        with tempfile.TemporaryDirectory() as d:
            res = analyze_file(self._write(d, "a4.wav", tone(440.0, decay=3.0), channels=2))
        self.assertTrue(res["pitched"])
        self.assertAlmostEqual(res["pitch_midi"], 69.0, delta=0.1)
        self.assertTrue(res["dual_mono"])
        self.assertEqual(res["channels"], 2)
        self.assertEqual(res["env"], "decaying")

    def test_low_note(self):
        with tempfile.TemporaryDirectory() as d:
            res = analyze_file(self._write(d, "e1.wav", tone(41.2, dur=2.0)))
        self.assertTrue(res["pitched"])
        self.assertAlmostEqual(res["pitch_midi"], 28.0, delta=0.15)

    def test_noise_burst_is_unpitched_percussive(self):
        rng = np.random.default_rng(1)
        t = np.arange(SR) / SR
        noise = (rng.standard_normal(SR) * np.exp(-t * 40.0) * 0.3).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            res = analyze_file(self._write(d, "hat.wav", noise))
        self.assertFalse(res["pitched"])
        self.assertEqual(res["env"], "percussive")

    def test_sustained_and_cut_ending(self):
        with tempfile.TemporaryDirectory() as d:
            res = analyze_file(self._write(d, "pad.wav", tone(220.0, dur=1.5)))
        self.assertEqual(res["env"], "sustained")
        self.assertTrue(res["ends_cut"])

    def test_slow_decay_longer_than_window_is_not_sustained(self):
        with tempfile.TemporaryDirectory() as d:
            res = analyze_file(self._write(d, "gong.wav", tone(220.0, dur=12.0, decay=0.5)))
        self.assertEqual(res["env"], "long-decay")
        self.assertFalse(res["ends_cut"])

    def test_true_tail_matches_last_frames(self):
        with tempfile.TemporaryDirectory() as d:
            data = tone(330.0, dur=2.5)
            path = self._write(d, "t.wav", data)
            with sf.SoundFile(path) as f:
                tail = features._true_tail(f, SR, 4410)
        self.assertEqual(tail.shape, (4410, 1))
        self.assertTrue(np.allclose(tail[:, 0], data[-4410:], atol=1e-5))

    def test_non_finite_values_become_none(self):
        out = features._finite({"a": float("nan"), "b": float("inf"), "c": 1.5, "d": "x", "e": 3})
        self.assertEqual(out, {"a": None, "b": None, "c": 1.5, "d": "x", "e": 3})

    def test_corrupt_file_reports_error(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.wav"
            bad.write_bytes(b"not audio at all" * 10)
            res = analyze_file(str(bad))
        self.assertIn("error", res)


if __name__ == "__main__":
    unittest.main()
