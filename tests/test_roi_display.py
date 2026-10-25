"""Rendering regression checks without loading model weights or starting Flask.

Run: python -m unittest discover -s tests -v
"""
import ast
import datetime
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np


class RoiDisplayTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / 'core' / 'ai_detector.py'
        tree = ast.parse(source.read_text(encoding='utf-8-sig'))
        # Execute the actual rendering function, excluding module-level model loading.
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'process_and_draw')
        self.config = SimpleNamespace(
            roi_polygon=[{'x': .4, 'y': .4}, {'x': .8, 'y': .4},
                         {'x': .8, 'y': .8}, {'x': .4, 'y': .8}],
            selected_classes=['car', 'train'], train_mode='no_train', dwell_time=5)
        self.env = dict(cv2=cv2, np=np, datetime=datetime, config=self.config,
                        active_objects={}, completed_events=[], current_date_str='',
                        roi_dwell_tracker={}, is_recording=False, video_writer=None,
                        pre_roll_buffer=deque(maxlen=90), last_activity_time=0,
                        TIMEOUT_SECONDS=3, MIN_EVENT_DURATION=1, MIN_CONFIRM_SECONDS=.5,
                        draw_roi=lambda frame: frame, translate_class=lambda name: name,
                        generate_html=lambda: '', format_time_str=lambda *a: '00:00:00',
                        update_txt_file=Mock(), remove_active_txt_entry=Mock())
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), self.env)

    def render(self, xyxy, name='car', now=0):
        coords = Mock()
        coords.cpu.return_value.numpy.return_value = np.array(xyxy)
        box = SimpleNamespace(cls=0, conf=.9, id=[7], xyxy=[coords])
        results = [SimpleNamespace(names={0: name}, boxes=[box])]
        with patch.object(cv2, 'rectangle', wraps=cv2.rectangle) as rectangles, \
                patch.object(cv2, 'putText', wraps=cv2.putText) as texts:
            output = self.env['process_and_draw'](
                np.zeros((100, 100, 3), np.uint8), results, 30, current_time=now,
                current_dt=datetime.datetime(2026, 1, 1))
        labels = [c.args[1] for c in texts.call_args_list if 'ID:' in c.args[1]]
        return rectangles.call_count, labels, output[2]

    def test_outside_box_and_id_hidden(self):
        self.assertEqual(self.render([5, 5, 20, 20]), (0, [], False))

    def test_inside_and_partial_overlap_visible(self):
        for box in ([45, 45, 60, 60], [30, 45, 45, 60]):
            with self.subTest(box=box):
                count, labels, danger = self.render(box)
                self.assertEqual(count, 2)
                self.assertIn('ID: 7', labels[0])
                self.assertFalse(danger)

    def test_no_roi_hides_overlays(self):
        self.config.roi_polygon = []
        self.assertEqual(self.render([45, 45, 60, 60]), (0, [], False))

    def test_train_outside_hidden_inside_visible_without_alarm(self):
        self.assertEqual(self.render([5, 5, 20, 20], 'train'), (0, [], False))
        count, labels, danger = self.render([45, 45, 60, 60], 'train')
        self.assertEqual(count, 2)
        self.assertEqual(len(labels), 1)
        self.assertFalse(danger)

    def test_exit_hides_immediately_and_expired_event_still_closes(self):
        self.render([45, 45, 60, 60])
        self.assertEqual(self.render([5, 5, 20, 20], now=2), (0, [], False))
        self.assertNotIn('7', self.env['roi_dwell_tracker'])
        self.env['active_objects']['7'] = dict(
            name='car', start_time_str='00:00:00', enter_time=0, last_seen=2,
            mode='no_train')
        self.render([5, 5, 20, 20], now=6)
        self.assertNotIn('7', self.env['active_objects'])
        self.assertEqual(len(self.env['completed_events']), 1)
        self.env['update_txt_file'].assert_called_once()


if __name__ == '__main__':
    unittest.main()
