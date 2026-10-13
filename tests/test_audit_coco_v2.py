"""Check independent COCO disagreement logic; never auto-overwrite ground truth."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import audit_coco_v2 as audit


class AuditCOCOTests(unittest.TestCase):
    def test_matching_original_bottle_is_not_a_missing_label(self):
        known=[(9,(10,10,110,110))]
        detected=[(39,"bottle",.85,(15,15,105,105))]
        self.assertEqual(audit.find_discrepancies(known,detected),[])

    def test_missing_bottle_and_utensil_are_review_candidates(self):
        known=[(8,(10,10,110,110))]
        detected=[(39,"bottle",.90,(15,15,105,105)),
                  (43,"knife",.78,(200,200,290,230))]
        candidate=audit.find_discrepancies(known,detected)
        self.assertEqual([c[0] for c in candidate],[9,3])
        self.assertEqual(known,[(8,(10,10,110,110))])

    def test_non_coco_classes_are_not_claimed_checked(self):
        known=[]
        detections=[(41,"cup",.9,(0,0,40,50)),
                    (0,"person",.9,(0,0,40,50))]
        self.assertEqual(audit.find_discrepancies(known,detections),[])


if __name__=="__main__":
    unittest.main()
