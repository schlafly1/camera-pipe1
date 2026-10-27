#!/usr/bin/env python3
"""
tools/test_phase2_reid.py — Offline unit tests for Phase 2 Vehicle Re-ID worker,
SQLite schema (vehicles, passes, sightings), multi-camera pass grouping, and
VLM attribute gating.
"""

import json
import os
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vehicle_reid_worker as vrw


def _make_unit_vec(seed, dim=2048):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def _blend_vec(base_vec, target_sim, seed, dim=2048):
    """Create a unit vector with exact cosine similarity `target_sim` to `base_vec`."""
    b = np.asarray(base_vec, dtype=np.float32)
    b /= np.linalg.norm(b)
    rng = np.random.default_rng(seed)
    ortho = rng.normal(size=dim).astype(np.float32)
    ortho = ortho - np.dot(ortho, b) * b
    ortho /= np.linalg.norm(ortho)
    out = float(target_sim) * b + float((1.0 - target_sim ** 2) ** 0.5) * ortho
    out /= np.linalg.norm(out)
    return [round(float(x), 6) for x in out.tolist()]


class TestPhase2Reid(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test_vehicles.db")
        self.worker = vrw.VehicleReidWorker(
            db_path=self.db_path,
            reid_url="http://127.0.0.1:9",
            update_chroma=False,
        )

    def tearDown(self):
        self.worker.conn.close()
        self.tmp.cleanup()

    def test_1_schema_and_tables(self):
        tables = {
            r[0]
            for r in self.worker.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        self.assertTrue({"vehicles", "passes", "sightings"}.issubset(tables))

    def test_2_attribute_gating_and_conflicts(self):
        # Compatible colors & bodies
        self.assertTrue(vrw.colors_compatible("silver", "gray"))
        self.assertTrue(vrw.colors_compatible("dark gray", "black"))
        self.assertFalse(vrw.colors_compatible("white", "black"))
        self.assertFalse(vrw.colors_compatible("red", "silver"))

        # Hard body conflicts
        self.assertFalse(vrw.body_types_compatible("sedan", "pickup"))
        self.assertFalse(vrw.body_types_compatible("sedan", "truck"))
        self.assertTrue(vrw.body_types_compatible("suv", "crossover"))

        # Roof trim conflict (Phase 0 key finding: white SUV w/ white roof vs white crossover w/ black roof)
        a_white_roof = {"color": "white", "body_type": "suv", "roof_trim": "white roof", "make_model": "unknown"}
        a_black_roof = {"color": "white", "body_type": "crossover", "roof_trim": "black roof", "make_model": "unknown"}
        self.assertTrue(vrw.roof_trim_conflicts("white roof", "black roof"))
        self.assertFalse(vrw.attrs_compatible(a_white_roof, a_black_roof))
        self.assertFalse(vrw.attrs_strongly_match(a_white_roof, a_black_roof))

    def test_3_multi_cam_pass_and_historical_familiarity(self):
        v_car1 = _make_unit_vec(101)
        emb1 = [round(float(x), 6) for x in v_car1.tolist()]
        emb1_cam3 = _blend_vec(v_car1, 0.52, seed=102)   # Cross-cam in same 60s pass
        emb1_pass2 = _blend_vec(v_car1, 0.65, seed=103)  # Later pass 2 hours later

        attrs1 = {
            "color": "silver",
            "body_type": "sedan",
            "roof_trim": "none",
            "make_model": "Toyota Corolla",
            "plate_partial": "",
            "motion": "moving",
            "label": "Silver Toyota Corolla sedan",
        }

        # Pass 1, sighting 1 on cam5 at t=1000
        m1 = self.worker.match_or_assign("car", "cam5", 1000.0, "2026-10-27T07:37:18", emb1, attrs1, "in")
        self.worker.conn.execute("INSERT INTO sightings (doc_id, pass_id, vehicle_id, timestamp_s, cam, cls, crop_w, crop_h, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", ("d1", m1["pass_id"], m1["vehicle_id"], 1000.0, "cam5", "car", 600, 400, json.dumps(emb1)))
        self.worker._update_vehicle_rollups(m1["vehicle_id"], 1000.0, "2026-10-27T07:37:18", "cam5", "c1.jpg", 600, 400, emb1, attrs1, m1["new_pass"])
        self.worker.conn.commit()
        self.assertEqual(m1["vehicle_id"], "veh_001")
        self.assertEqual(m1["is_unfamiliar"], 1)
        self.assertEqual(m1["match_method"], "new")

        # Pass 1, sighting 2 on cam3 at t=1003.2 (3.2s later -> same pass!)
        m2 = self.worker.match_or_assign("car", "cam3", 1003.2, "2026-10-27T07:37:21", emb1_cam3, attrs1, "in")
        self.worker._update_vehicle_rollups(m2["vehicle_id"], 1003.2, "2026-10-27T07:37:21", "cam3", "c2.jpg", 900, 450, emb1_cam3, attrs1, m2["new_pass"])
        self.worker.conn.commit()
        self.assertEqual(m2["vehicle_id"], "veh_001")
        self.assertEqual(m2["pass_id"], m1["pass_id"])
        self.assertEqual(m2["match_method"], "cross_cam_pass")

        v_row = self.worker.conn.execute("SELECT * FROM vehicles WHERE vehicle_id='veh_001'").fetchone()
        self.assertEqual(v_row["pass_count"], 1)
        self.assertEqual(v_row["sighting_count"], 2)
        self.assertEqual(v_row["is_familiar"], 0)
        self.assertEqual(json.loads(v_row["cameras_seen"]), ["cam5", "cam3"])

        # Pass 2 of same car 2 hours later (t=8200) -> historical match, becomes familiar!
        m3 = self.worker.match_or_assign("car", "cam4", 8200.0, "2026-10-27T09:37:18", emb1_pass2, attrs1, "street")
        self.worker._update_vehicle_rollups(m3["vehicle_id"], 8200.0, "2026-10-27T09:37:18", "cam4", "c3.jpg", 700, 350, emb1_pass2, attrs1, m3["new_pass"])
        self.worker.conn.commit()
        self.assertEqual(m3["vehicle_id"], "veh_001")
        self.assertNotEqual(m3["pass_id"], m1["pass_id"])
        self.assertEqual(m3["is_unfamiliar"], 0)
        self.assertEqual(m3["match_method"], "reid_high")

        v_row2 = self.worker.conn.execute("SELECT * FROM vehicles WHERE vehicle_id='veh_001'").fetchone()
        self.assertEqual(v_row2["pass_count"], 2)
        self.assertEqual(v_row2["sighting_count"], 3)
        self.assertEqual(v_row2["is_familiar"], 1)

    def test_4_gray_zone_vlm_gate_blocks_conflicting_vehicle(self):
        v_base = _make_unit_vec(201)
        emb_base = [round(float(x), 6) for x in v_base.tolist()]
        emb_gray = _blend_vec(v_base, 0.55, seed=202)  # In gray zone [0.48, 0.62)

        attrs_white_roof = {"color": "white", "body_type": "suv", "roof_trim": "white roof", "make_model": "unknown"}
        attrs_black_roof = {"color": "white", "body_type": "crossover", "roof_trim": "black roof", "make_model": "unknown"}

        m1 = self.worker.match_or_assign("car", "cam4", 2000.0, "2026-10-27T08:00:00", emb_base, attrs_white_roof, "street")
        self.worker._update_vehicle_rollups(m1["vehicle_id"], 2000.0, "2026-10-27T08:00:00", "cam4", "w1.jpg", 500, 300, emb_base, attrs_white_roof, m1["new_pass"])
        self.worker.conn.commit()

        # 10 minutes later: gray-zone similarity (0.55) but conflicting roof trim -> MUST NOT MERGE
        m2 = self.worker.match_or_assign("car", "cam4", 2600.0, "2026-10-27T08:10:00", emb_gray, attrs_black_roof, "street")
        self.assertNotEqual(m2["vehicle_id"], m1["vehicle_id"])
        self.assertEqual(m2["is_unfamiliar"], 1)
        self.assertEqual(m2["match_method"], "new")


if __name__ == "__main__":
    unittest.main(verbosity=2)
