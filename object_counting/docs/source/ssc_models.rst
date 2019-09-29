************
SSC Variants
************

This module provides the implementation of the possible variants of the SSC counter:

- SSC300 Basic
- SSC300 Merged

For each of the 2 versions, there are 2 possible additions:

- Condensed Predictors (condense the predictors for different scales into one prediction with an additional layer)
- LSTM (before the layer to condense the predictors, add an lstm cell to incorporate the context among the different scales)

SSC300 Basic
============

.. automodule:: object_counting.keras_ssc.models.keras_ssc300
   :members:

SSC300 Merged
=============

.. automodule:: object_counting.keras_ssc.models.keras_ssc300_4_merged
   :members:

