"""No LLM requests. Guard legal plans, equivalent context, and stale references."""
import json
import unittest
from common import check_plan, make_prompt


class ProtocolTests(unittest.TestCase):
    names = ['table_office_001','lamp_office_001','table_dining_001']
    rooms = ['office_floor','dining_floor']
    catalog = ['lamp','book','mug','vase','sofa']

    def valid(self):
        return dict(status='execute',delete=[],insert=[dict(asset='vase',instance_key='vase_002',
                    on_instance='table_office_001',in_room='office_floor',place_strategy='top')],message='ok')

    def test_valid(self):
        p = self.valid()
        self.assertIs(check_plan(p,self.names,self.rooms,self.catalog),p)

    def test_missing_deletion_rejected(self):
        p = self.valid()
        p['delete'] = ['already_deleted_mug']
        with self.assertRaises(ValueError):
            check_plan(p,self.names,self.rooms,self.catalog)

    def test_clarification_cannot_edit(self):
        p = self.valid()
        p['status'] = 'clarify'
        with self.assertRaises(ValueError):
            check_plan(p,self.names,self.rooms,self.catalog)

    def test_no_overwrite(self):
        p = self.valid()
        p['insert'][0]['instance_key'] = 'lamp_office_001'
        with self.assertRaises(ValueError):
            check_plan(p,self.names,self.rooms,self.catalog)

    def test_no_unsafe_paths_or_new_algorithms(self):
        for key in ['../escape','floor_foo','wall_foo']:
            p = self.valid()
            p['insert'][0]['instance_key'] = key
            with self.assertRaises(ValueError):
                check_plan(p,self.names,self.rooms,self.catalog)
        p = self.valid()
        p['insert'][0]['scale'] = .1
        with self.assertRaises(ValueError):
            check_plan(p,self.names,self.rooms,self.catalog)

    def test_context_keeps_grounded_history_and_tombstone(self):
        inserted = dict(asset='mug',instance_key='mug_a',on_instance='table_office_001',in_room='office_floor',place_strategy='top')
        history = [dict(instruction='在办公室桌上放一个杯子。',status='execute',deleted=[],inserted=[inserted]),
                   dict(instruction='删掉刚才的杯子。',status='execute',deleted=['mug_a'],inserted=[])]
        for c in ['M0','M1','M2','M3']:
            p = make_prompt(c,self.names,self.rooms,self.catalog,history,'再删掉刚才那个杯子。')
            for value in [*self.names,*self.rooms,*self.catalog,'mug_a','在办公室桌上放一个杯子。','再删掉刚才那个杯子。']:
                self.assertIn(value,p)
            self.assertIn('noop',p)

    def test_m2_prompt_identical_to_m0(self):
        a = (self.names,self.rooms,self.catalog,[],'删除办公室的台灯。')
        self.assertEqual(make_prompt('M0',*a),make_prompt('M2',*a))


if __name__ == '__main__':
    unittest.main()
