#!/usr/bin/env python3
"""
tools/test_phase3_ui_api.py — Unit & integration tests for Phase 3:
  - /api/vehicles (filters, search, sorting, summary stats)
  - /api/vehicles/{vehicle_id} (passes, sightings, similar_vehicles)
  - /api/vehicle_sightings (chronological log, unfamiliar_only, multi_cam_only, camera filter)
  - /api/vehicles/{vehicle_id}/update (custom_label preservation across subsequent sightings)
  - /api/vehicles/merge (pass/sighting reassignment, mean_embedding recomputation, merged_into exclusion)
  - /api/vehicles/split (splitting a pass or sighting into a new vehicle_id)
"""

import os
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vehicle_reid_worker as vrw
from fastapi.testclient import TestClient
import query_server as qs


def _unit_vec(seed, dim=64):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return vrw.l2_normalize(v)


def _perturb_vec(base, noise=0.05, seed=99):
    rng = np.random.RandomState(seed)
    v = np.asarray(base, dtype=np.float32) + rng.randn(len(base)).astype(np.float32) * noise
    return vrw.l2_normalize(v)


class FakeWorker(vrw.VehicleReidWorker):
    def __init__(self, db_path, mock_responses):
        super().__init__(db_path=db_path, reid_url="http://mock", update_chroma=False)
        self.mock_responses = mock_responses

    def call_reid_server(self, ev):
        return self.mock_responses[ev["doc_id"]]


class TestPhase3VehiclesAPI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test_vehicles.db")
        qs.VEHICLES_DB_PATH = self.db_path
        # Monkeypatch _get_vdb_worker in query_server so tests don't touch live ChromaDB
        self._orig_get_worker = qs._get_vdb_worker
        qs._get_vdb_worker = lambda: vrw.VehicleReidWorker(db_path=self.db_path, update_chroma=False)
        self.client = TestClient(qs.app)

        # Seed 3 vehicles across 4 passes and 5 sightings
        self.emb_a = _unit_vec(10)
        self.emb_b = _perturb_vec(self.emb_a, noise=0.35, seed=20)  # somewhat similar silver SUV
        self.emb_c = _unit_vec(30)

        mocks = {
            "d1": {
                "ok": True,
                "embedding": self.emb_a,
                "crop_shape": [300, 450],
                "crop_source": "main_crop_yolo",
                "attrs": {
                    "color": "maroon", "body_type": "pickup", "roof_trim": "none",
                    "make_model": "Toyota Tacoma", "plate_partial": "", "motion": "moving",
                    "label": "Maroon Toyota Tacoma pickup",
                },
            },
            "d2": {
                "ok": True,
                "embedding": _perturb_vec(self.emb_a, noise=0.04, seed=11),
                "crop_shape": [320, 480],
                "crop_source": "main_crop_yolo",
                "attrs": {
                    "color": "maroon", "body_type": "pickup", "roof_trim": "none",
                    "make_model": "Toyota Tacoma", "plate_partial": "", "motion": "moving",
                    "label": "Maroon Toyota Tacoma pickup",
                },
            },
            "d3": {
                "ok": True,
                "embedding": _perturb_vec(self.emb_a, noise=0.03, seed=12),
                "crop_shape": [340, 500],
                "crop_source": "main_crop_yolo",
                "attrs": {
                    "color": "maroon", "body_type": "pickup", "roof_trim": "none",
                    "make_model": "Toyota Tacoma", "plate_partial": "", "motion": "moving",
                    "label": "Maroon Toyota Tacoma pickup",
                },
            },
            "d4": {
                "ok": True,
                "embedding": self.emb_b,
                "crop_shape": [260, 380],
                "crop_source": "main_crop_yolo",
                "attrs": {
                    "color": "red", "body_type": "pickup", "roof_trim": "none",
                    "make_model": "Toyota Tacoma", "plate_partial": "", "motion": "moving",
                    "label": "Red Toyota Tacoma pickup",
                },
            },
            "d5": {
                "ok": True,
                "embedding": self.emb_c,
                "crop_shape": [240, 200],
                "crop_source": "main_crop_yolo",
                "attrs": {
                    "color": "red", "body_type": "mountain_bike", "roof_trim": "none",
                    "make_model": "unknown", "plate_partial": "", "motion": "moving",
                    "label": "Red mountain bike",
                },
            },
        }
        self.worker = FakeWorker(self.db_path, mocks)
        # Pass 1 of veh_001: multi-cam (cam6 -> cam3)
        self.worker.process_event_record({"doc_id": "d1", "evt_id": 1, "cam": "cam6", "cls": "car", "timestamp_s": 1000.0, "timestamp": "2026-10-27T08:00:00-07:00", "main_crop_path": "snapshots/vehicles/d1.jpg", "vlm_description": "A maroon Tacoma"})
        self.worker.process_event_record({"doc_id": "d2", "evt_id": 2, "cam": "cam3", "cls": "car", "timestamp_s": 1008.0, "timestamp": "2026-10-27T08:00:08-07:00", "main_crop_path": "snapshots/vehicles/d2.jpg", "vlm_description": "A maroon Tacoma on driveway"})
        # Pass 2 of veh_001: 1 hour later -> makes veh_001 familiar!
        self.worker.process_event_record({"doc_id": "d3", "evt_id": 3, "cam": "cam5", "cls": "car", "timestamp_s": 4600.0, "timestamp": "2026-10-27T09:00:00-07:00", "main_crop_path": "snapshots/vehicles/d3.jpg", "vlm_description": "Maroon Tacoma returning"})
        # Pass 1 of veh_002 (unfamiliar car)
        self.worker.process_event_record({"doc_id": "d4", "evt_id": 4, "cam": "cam6", "cls": "car", "timestamp_s": 5000.0, "timestamp": "2026-10-27T09:10:00-07:00", "main_crop_path": "snapshots/vehicles/d4.jpg", "vlm_description": "Red pickup on street"})
        # Pass 1 of veh_003 (unfamiliar bicycle)
        self.worker.process_event_record({"doc_id": "d5", "evt_id": 5, "cam": "cam4", "cls": "bicycle", "timestamp_s": 5500.0, "timestamp": "2026-10-27T09:20:00-07:00", "main_crop_path": "snapshots/vehicles/d5.jpg", "vlm_description": "Red mountain bike"})

    def tearDown(self):
        self.worker.conn.close()
        qs._get_vdb_worker = self._orig_get_worker
        self.tmp.cleanup()

    def test_list_vehicles_and_filters(self):
        r = self.client.get("/api/vehicles")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["summary"]["total_vehicles"], 3)
        self.assertEqual(data["summary"]["familiar_count"], 1)
        self.assertEqual(data["summary"]["unfamiliar_count"], 2)
        self.assertEqual(data["summary"]["total_passes"], 4)
        self.assertEqual(data["summary"]["total_sightings"], 5)

        # Unfamiliar filter
        r_unfam = self.client.get("/api/vehicles?filter=unfamiliar").json()
        self.assertEqual(r_unfam["count"], 2)
        self.assertTrue(all(v["is_familiar"] == 0 for v in r_unfam["vehicles"]))

        # Familiar filter
        r_fam = self.client.get("/api/vehicles?filter=familiar").json()
        self.assertEqual(r_fam["count"], 1)
        self.assertEqual(r_fam["vehicles"][0]["vehicle_id"], "veh_001")

        # Class filter
        r_bike = self.client.get("/api/vehicles?cls=bicycle").json()
        self.assertEqual(r_bike["count"], 1)
        self.assertEqual(r_bike["vehicles"][0]["vehicle_id"], "veh_003")

        # Search query
        r_q = self.client.get("/api/vehicles?q=tacoma").json()
        self.assertEqual(r_q["count"], 2)

    def test_get_vehicle_detail_and_similar(self):
        r = self.client.get("/api/vehicles/veh_001")
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(d["vehicle"]["vehicle_id"], "veh_001")
        self.assertEqual(len(d["passes"]), 2)
        self.assertEqual(len(d["sightings"]), 3)
        self.assertTrue(len(d["similar_vehicles"]) >= 1)
        self.assertEqual(d["similar_vehicles"][0]["vehicle_id"], "veh_002")

    def test_update_custom_label_preserved_on_new_sighting(self):
        r = self.client.post("/api/vehicles/veh_001/update", json={
            "label": "Neighbor's Maroon Tacoma",
            "notes": "Lives down the street",
        })
        self.assertEqual(r.status_code, 200)
        v = r.json()["vehicle"]
        self.assertEqual(v["label"], "Neighbor's Maroon Tacoma")
        self.assertEqual(v["custom_label"], 1)

        # Now process a new sighting of veh_001 and verify custom label is NOT overwritten
        self.worker.mock_responses["d6"] = {
            "ok": True,
            "embedding": _perturb_vec(self.emb_a, noise=0.02, seed=19),
            "crop_shape": [300, 450],
            "crop_source": "main_crop_yolo",
            "attrs": {
                "color": "maroon", "body_type": "pickup", "roof_trim": "none",
                "make_model": "Toyota Tacoma", "plate_partial": "", "motion": "moving",
                "label": "Generic Auto Label",
            },
        }
        self.worker.process_event_record({"doc_id": "d6", "evt_id": 6, "cam": "cam6", "cls": "car", "timestamp_s": 8000.0, "timestamp": "2026-10-27T10:00:00-07:00", "main_crop_path": "snapshots/vehicles/d6.jpg"})
        r2 = self.client.get("/api/vehicles/veh_001").json()
        self.assertEqual(r2["vehicle"]["label"], "Neighbor's Maroon Tacoma")
        self.assertEqual(r2["vehicle"]["sighting_count"], 4)

    def test_merge_and_split_vehicles(self):
        # Merge veh_002 into veh_001
        r = self.client.post("/api/vehicles/merge", json={
            "source_id": "veh_002",
            "target_id": "veh_001",
        })
        self.assertEqual(r.status_code, 200)
        merged = r.json()["vehicle"]
        self.assertEqual(merged["vehicle_id"], "veh_001")
        self.assertEqual(merged["pass_count"], 3)
        self.assertEqual(merged["sighting_count"], 4)

        # Active vehicles list should now have 2 vehicles (veh_001 and veh_003)
        r_list = self.client.get("/api/vehicles").json()
        self.assertEqual(r_list["summary"]["total_vehicles"], 2)

        # Now split sighting #4 (which came from d4) back out into a new vehicle
        d_veh1 = self.client.get("/api/vehicles/veh_001").json()
        d4_sight = [s for s in d_veh1["sightings"] if s["doc_id"] == "d4"][0]
        r_split = self.client.post("/api/vehicles/split", json={
            "sighting_ids": [d4_sight["sighting_id"]],
            "target_vehicle_id": "new",
        })
        self.assertEqual(r_split.status_code, 200)
        new_veh = r_split.json()["target_vehicle"]
        self.assertEqual(new_veh["vehicle_id"], "veh_004")
        self.assertEqual(new_veh["sighting_count"], 1)
        self.assertEqual(new_veh["pass_count"], 1)

        # Verify veh_001 is back to 2 passes and 3 sightings
        d_veh1_after = self.client.get("/api/vehicles/veh_001").json()["vehicle"]
        self.assertEqual(d_veh1_after["pass_count"], 2)
        self.assertEqual(d_veh1_after["sighting_count"], 3)

    def test_vehicle_sightings_log_filters(self):
        r_all = self.client.get("/api/vehicle_sightings").json()
        self.assertEqual(r_all["count"], 5)

        r_multi = self.client.get("/api/vehicle_sightings?multi_cam_only=1").json()
        self.assertEqual(r_multi["count"], 2)  # d1 and d2 in pass #1

        r_unfam = self.client.get("/api/vehicle_sightings?unfamiliar_only=1").json()
        self.assertEqual(r_unfam["count"], 2)  # d4 and d5


if __name__ == "__main__":
    unittest.main()
