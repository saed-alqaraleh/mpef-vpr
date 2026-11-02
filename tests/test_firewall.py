import subprocess
import sys
import unittest


class FirewallTest(unittest.TestCase):
    def test_production_tree_respects_firewall(self):
        result = subprocess.run([sys.executable, "scripts/check_firewall.py"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
