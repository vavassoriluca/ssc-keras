## Count Models

The models provided by this module are the followings:

1. SSC: Single-Shot Multiscale Counter
2. Chattopadhyay's Counting Everyday Objects algorithms
    - Glance
    - Associative Subitizing
    - Sequential Subitizing

In the *tutorials* directory, several notebooks explain how to use the different models and configure their parameters. A complete documentation is provided in the directory folder. It has to be built with the `sphinx-build` command.

## Performance table

The lower the mRMSE the better.

|   Counter   | VOC07 (mRMSE) | FPS | # Parameters |
|:------------:|:-------------------:|:-------------------:|:----------:|
|  SSC-basic   |         0.39        |          20.0       |      39.3 M     |
|     SSC-m    |         0.36        |          18.8       |     130.7 M     |
|    SSC-m-c   |         0.35        |          18.7       |     130.7 M     |
| SSC-m-lstm-c |         0.35        |          16.4       |     130.7 M     |
|  aso-sub-2l  |         0.43        |          0.60       |      59.7 M     |
| seq-sub-noft |         0.42        |          0.79       |     100.5 M     |

