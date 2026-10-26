from copy import deepcopy
import unittest

from scripts.rebalance_coco_validation import balanced_quotas,relative_areas,spaced_selection,statistics


class RebalanceValidationTests(unittest.TestCase):
    def test_balanced_quotas_redistribute_small_sequence_capacity(self):
        self.assertEqual(balanced_quotas({'a':1,'b':9,'c':5},10),{'a':1,'b':5,'c':4})
        with self.assertRaises(ValueError):balanced_quotas({'a':2},3)

    def test_selection_prefers_no_small_companions_and_is_deterministic(self):
        rows=[dict(image={'file_name':f'frame_{frame}.jpg'},frame=frame,all_boxes_large=pure)
              for frame,pure in [(0,False),(10,True),(20,True),(30,False)]]
        before=deepcopy(rows)
        picked=spaced_selection(rows,2,42)
        self.assertTrue(all(r['all_boxes_large'] for r in picked))
        self.assertEqual(picked,spaced_selection(rows,2,42))
        self.assertEqual(rows,before)

    def test_relative_sizes_use_native_image_area_and_keep_companion_boxes(self):
        image=dict(id=1,file_name='seq_0001.jpg',width=1000,height=1000)
        annotations=[dict(image_id=1,bbox=[0,0,200,200]),dict(image_id=1,bbox=[400,400,20,20])]
        self.assertEqual(relative_areas(image,annotations),[.04,.0004])
        result=statistics(dict(images=[image,dict(id=2,file_name='seq_0002.jpg',width=1000,height=1000)],annotations=annotations),.02)
        self.assertEqual(result['positive_image_count'],1)
        self.assertEqual(result['negative_image_count'],1)
        self.assertEqual(result['box_count'],2)
        self.assertEqual(result['large_target_boxes'],1)
        self.assertEqual(result['images_with_large_target'],1)
