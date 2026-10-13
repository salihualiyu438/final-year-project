Model 2
=======

Place the separately trained weapon/threat detector weight here as:

    models\model2.pt

Expected Model 2 class IDs:
    0 = large_gun
    1 = short_gun
    2 = matchet/knife

Model 2 is independent of Model 1. If model2.pt is absent, the application
continues to run using Model 1 only.
