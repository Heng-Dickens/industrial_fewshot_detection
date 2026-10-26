import asyncio
import json
import unittest
import industrial_fewshot.detectors.mllm as target
import industrial_fewshot.legacy.batch_annotate as batch_annotate
from tests.support import LABEL, FakeClient, response

_MISSING = object()


def make_object(flag=_MISSING) -> dict:
    item = {"label": LABEL, "bbox_2d": [1, 2, 3, 4]}
    if flag is not _MISSING:
        item["hanging_object_confirmed"] = flag
    return item



class SuspensionGateTest(unittest.TestCase):
    CONFIG_ON = {"require_suspension_evidence": True}
    CONFIG_OFF = {"require_suspension_evidence": False}

    def prediction(self, objects, target_count=1, status="annotated"):
        return {
            "annotation_status": status,
            "reason_code": "target_detected" if status == "annotated" else "no_target",
            "reason": "第一遍判断",
            "target_count": target_count,
            "objects": objects,
        }

    def test_drops_only_unconfirmed_objects(self):
        gated, info = target.apply_suspension_gate(
            self.prediction([make_object(True), make_object(False)]), self.CONFIG_ON
        )
        self.assertEqual(len(gated["objects"]), 1)
        self.assertEqual(gated["target_count"], 1)
        self.assertEqual(gated["annotation_status"], "annotated")
        self.assertEqual(info["dropped"], 1)
        self.assertEqual(info["missing_flag"], 0)

    def test_dropping_everything_downgrades_to_no_target(self):
        gated, info = target.apply_suspension_gate(
            self.prediction([make_object(False)]), self.CONFIG_ON
        )
        self.assertEqual(gated["objects"], [])
        self.assertEqual(gated["target_count"], 0)
        self.assertEqual(gated["annotation_status"], "no_target")
        self.assertEqual(gated["reason_code"], "no_target")
        self.assertIn("已丢弃", gated["reason"])
        self.assertEqual(info["dropped"], 1)

    def test_missing_flag_is_kept_and_counted(self):
        gated, info = target.apply_suspension_gate(
            self.prediction([make_object()]), self.CONFIG_ON
        )
        self.assertEqual(len(gated["objects"]), 1)
        self.assertEqual(info["dropped"], 0)
        self.assertEqual(info["missing_flag"], 1)

    def test_non_boolean_flag_is_kept_and_counted(self):
        gated, info = target.apply_suspension_gate(
            self.prediction([make_object("false")]), self.CONFIG_ON
        )
        self.assertEqual(len(gated["objects"]), 1)
        self.assertEqual(info["dropped"], 0)
        self.assertEqual(info["missing_flag"], 1)

    def test_no_drop_leaves_target_count_untouched(self):
        # target_count 与 objects 数量不一致时不能被悄悄修正，
        # 否则 validate_prediction 会失去这个有效的报错信号
        gated, info = target.apply_suspension_gate(
            self.prediction([make_object(True)], target_count=5), self.CONFIG_ON
        )
        self.assertEqual(gated["target_count"], 5)
        self.assertEqual(info["dropped"], 0)

    def test_disabled_switch_passes_everything_through(self):
        prediction = self.prediction([make_object(False)])
        gated, info = target.apply_suspension_gate(prediction, self.CONFIG_OFF)
        self.assertIs(gated, prediction)
        self.assertEqual(info["dropped"], 0)
        self.assertFalse(info["enabled"])

    def test_malformed_objects_pass_through(self):
        for objects in ("nope", [1, 2], None):
            prediction = self.prediction(objects)
            gated, info = target.apply_suspension_gate(prediction, self.CONFIG_ON)
            self.assertIs(gated, prediction)
            self.assertEqual(info["dropped"], 0)



class JsonSchemaGateTest(unittest.TestCase):
    def item_schema(self, config):
        schema = batch_annotate.build_json_schema(config)
        return schema["properties"]["objects"]["items"]

    def test_closed_switch_keeps_original_object_schema(self):
        item = self.item_schema({"labels": [{"name": LABEL}]})
        self.assertEqual(item["required"], ["label", "bbox_2d"])
        self.assertNotIn("hanging_object_confirmed", item["properties"])

    def test_open_switch_requires_boolean_confirmation(self):
        item = self.item_schema(
            {"labels": [{"name": LABEL}], "require_suspension_evidence": True}
        )
        self.assertIn("hanging_object_confirmed", item["required"])
        self.assertEqual(
            item["properties"]["hanging_object_confirmed"], {"type": "boolean"}
        )
        self.assertFalse(item["additionalProperties"])


class ModelRequestTest(unittest.TestCase):
    def config(self, gate=False):
        return {
            "model": "test-model", "labels": [{"name": "target"}],
            "temperature": 0.1, "top_p": 0.9, "max_tokens": 512,
            "stream": False, "enable_thinking": False,
            "structured_output_mode": "json_schema", "require_confidence": True,
            "require_suspension_evidence": gate,
        }

    def request(self, payload, gate=False):
        config = self.config(gate)
        client = FakeClient([payload])
        result = asyncio.run(target.request_annotation(
            client, config, "EXTERNAL_RULES", [], "FULL_IMAGE",
            target.build_response_format(config), {"target"}, 1000, 500,
        ))
        return result, client

    def test_shared_request_preserves_score_and_full_image_coordinates(self):
        result, client = self.request(response())
        self.assertEqual(result["annotation"]["objects"][0]["confidence"], 0.9)
        self.assertEqual(result["annotation"]["objects"][0]["bbox_pixels_xyxy"], [100, 100, 800, 450])
        self.assertEqual(client.requests[0]["messages"][0]["content"][-1]["image_url"]["url"], "FULL_IMAGE")

    def test_confirmed_false_is_filtered_only_when_gate_is_enabled(self):
        payload = json.loads(response())
        payload["objects"][0]["hanging_object_confirmed"] = False
        for enabled in (False, True):
            result, _ = self.request(json.dumps(payload), enabled)
            self.assertEqual(len(result["annotation"]["objects"]), 0 if enabled else 1)

    def test_empty_response_is_a_failure(self):
        with self.assertRaisesRegex(ValueError, "响应内容为空"):
            self.request("")
