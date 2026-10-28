import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from run_chache_qwen27b_full5shot_top3 import rank_top3, DEFAULT_VARIANTS


def groups():
    return [dict(experiment=dict(positive_count=5,negative_count=0,variant=v),
                 evaluation_status='complete',failed_request_count=0,
                 metrics=dict(mAP50=.1+i*.1,mAP50_95=.1,recall=.5,f2=.5,precision=1-i*.1))
            for i,v in enumerate(DEFAULT_VARIANTS)]


class TopThreeSelectionTest(unittest.TestCase):
    def test_ranks_macro_ap50_excludes_text_and_rejects_missing_or_failed_results(self):
        data=groups()
        text=copy.deepcopy(data[0]);text['experiment']['positive_count']=0;text['metrics']['mAP50']=1
        selected=rank_top3([*data,text])
        self.assertEqual([g['experiment']['variant'] for g in selected],list(reversed(DEFAULT_VARIANTS[-3:])))
        with self.assertRaises(ValueError): rank_top3(data[:-1])
        data[-1]['failed_request_count']=1
        with self.assertRaises(ValueError): rank_top3(data)

    def test_ties_use_ap5095_then_recall_then_f2_then_variant_name(self):
        data=groups()
        for g in data:g['metrics'].update(mAP50=.5,mAP50_95=.2,recall=.5,f2=.5)
        data[2]['metrics']['mAP50_95']=.3
        data[3]['metrics']['recall']=.6
        data[4]['metrics']['f2']=.6
        expected=[data[i]['experiment']['variant'] for i in [2,3,4]]
        self.assertEqual([g['experiment']['variant'] for g in rank_top3(data)],expected)
        for g in data:g['metrics'].update(mAP50_95=.2,recall=.5,f2=.5)
        self.assertEqual([g['experiment']['variant'] for g in rank_top3(data)],sorted(DEFAULT_VARIANTS)[:3])


if __name__=='__main__':unittest.main()
