***********
SSC Encoder
***********

This module provides the methods to encode the ground truth to train SSC.
The anchor boxes, as in SSD, are matched to the anchor box generated a priori that has the highest IoU (intersection over union) with the true box.
Than the matched are summed to get the count of each class assigned to a specific predictor.
The :class:`object_counting.keras_ssc.ssc_encoder.ssc_input_encoder_1pred.SSCInputEncoder1Pred` version encode the count globally, without splitting it  by scale.

.. automodule:: object_counting.keras_ssc.ssc_encoder.ssc_input_encoder
   :members:

.. automodule:: object_counting.keras_ssc.ssc_encoder.ssc_input_encoder_1pred
   :members:

