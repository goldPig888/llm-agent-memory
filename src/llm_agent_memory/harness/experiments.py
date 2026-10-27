"""Opt-in harness ablations; no object labels, poses, or automatic robot actions."""
import ast
from dataclasses import dataclass, field
import json


VARIANTS = {
    'control': dict(recovery=False, clean_images=False),
    'recovery': dict(recovery=True, clean_images=False),
    'clean-images': dict(recovery=False, clean_images=True),
    'combined': dict(recovery=True, clean_images=True),
}

# Opt-in visual experiments retain the same prompt, crop, geometry, and checker.
VARIANTS['clean-bright'] = dict(recovery=False, clean_images=True, image_tone='gamma-0.8')
VARIANTS['clean-contrast'] = dict(recovery=False, clean_images=True, image_tone='autocontrast-1pct')


def tone_image(image, mode='identity'):
    from PIL import ImageOps
    if mode == 'identity':
        return image
    if mode == 'gamma-0.8':
        # Monotonic, fixed RGB transform. No per-object processing or new detail.
        lut=[round(255*(x/255)**.8) for x in range(256)]
        return image.point(lut*3)
    if mode == 'autocontrast-1pct':
        return ImageOps.autocontrast(image,cutoff=1,preserve_tone=True)
    raise ValueError('Unknown image tone: '+mode)


def image_instructions(text, clean):
    if not clean:
        return text
    # Explicit final override covers inherited overlay descriptions without
    # changing pixel coordinates, crop, or the number of model images.
    return text + '''\nIMAGE OVERRIDE FOR THIS EXPERIMENT: both images contain clean RGB with
NO drawn grid, labels, reticles, jaw bars, or magenta lines. Do not look for
these overlays. Use visual object pixels and camera-specific deprojection.
look_at changes only the agentview crop; camera coordinates remain calibrated.
'''


def canonical(code, image_context=None):
    try:
        tree=ast.parse(code)
        key=ast.dump(tree, include_attributes=False)
        # The same pixel program can select a different world target after a
        # crop change or a physics step. World-coordinate retries remain
        # remembered across observation-only crop changes.
        if any(isinstance(node,ast.Attribute) and node.attr in {'deproject','deproject_depth'}
               for node in ast.walk(tree)):
            key+=repr(image_context)
        return key
    except SyntaxError as exc:
        raise ValueError('Invalid plan syntax') from exc


@dataclass
class RecoveryMonitor:
    """Remembers rejected programs only while the measured robot pose is unchanged."""
    pose: object = None
    rejected: dict = field(default_factory=dict)
    stagnant: int = 0
    last_status: str = 'Ready'

    def sync(self, pose):
        import numpy as np
        pose = np.asarray(pose, float)
        if self.pose is None or np.max(np.abs(pose-self.pose)) > .002:
            self.rejected.clear()
        self.pose = pose.copy()

    def before(self, code, pose, image_context=None):
        self.sync(pose)
        key = canonical(code, image_context)
        if key in self.rejected:
            return json.dumps(dict(status='repeat_rejected', previous_status=self.rejected[key],
                message='Same program already failed at this measured pose. Nothing executed. '
                'Choose a DIFFERENT endpoint, smaller displacement, different orientation, or route. '
                'An identical retry will not be executed.'))
        return None

    def after(self, code, feedback, start_step, end_step, pose, image_context=None):
        self.sync(pose)
        try:
            status = json.loads(feedback).get('status', 'plan_error')
        except (ValueError, TypeError):
            status = 'plan_error'
        self.last_status = status
        self.stagnant = self.stagnant+1 if end_step == start_step else 0
        if code and status in {'IK_FAILURE','COLLISION','blocked','CARRY_UNVERIFIED'}:
            self.rejected[canonical(code,image_context)] = status

    def advice(self):
        if not self.rejected and self.stagnant < 2:
            return None
        return dict(last_status=self.last_status, decisions_without_motion=self.stagnant,
            rejected_programs_at_current_pose=len(self.rejected),
            instruction='Use the measured eef_position. Rejected segments have not moved the arm. '
            'Do not alternate unchanged camera views with the same failing motion. '
            'For IK_FAILURE change endpoint or orientation; try a smaller 1–2 cm '
            'displacement preserving orientation. For COLLISION change the route, '
            'not just speed. Re-observing an unchanged pose does not fix IK. '
            'Every recovery still passes the same motion checker.')
