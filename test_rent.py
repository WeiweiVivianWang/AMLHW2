#!/usr/bin/env python
"""A boilerplate script to be customized for data projects.

This script-level docstring will double as the description when the script is
called with the --help or -h option.

"""

from homework2_rent import score_rent

def test_rent():
  """Checks the R^2 returned by score_rent()"""
  assert score_rent() >= 0.9