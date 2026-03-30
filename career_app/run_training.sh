#!/bin/bash
python3 step1_preprocess.py && python3 step2_train_lgbm.py && python3 step3_train_xgb.py && python3 step4_train_cat.py && python3 step5_stack.py
