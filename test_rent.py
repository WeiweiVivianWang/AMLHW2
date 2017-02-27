#!/usr/bin/env python
"""test_rent.py
Checks that the R^2 of the model is greater than or equal to 0.5.
"""

from homework2_rent import *

def test_rent():
  """Ensures that the R^2 of the model meets the expected outcome"""
  assert score_rent(model) >= 0.5