Key Files Summary:

  1. check_lsl_streams.py - Verify LSL
  connections
  2. simulation/eeg_model_integration.py --continuous
  3. simulation/run_simulation_mi.py --mode real_eeg
  4. eeg_cumulative_control.txt - Command


  Dependencies Required:

  - pylsl - For LSL communication
  - pygame - For the game
  - numpy, scipy - For signal processing
  - scikit-learn - For classification
  - Trained MI model in MI/models/22



  python simulation/eeg_model_integration_v2.py --phase phase1 --participant T-001

  pre-training event logger is saved in BCI_Wheelchair_Project\pre_training_event_log, we need to merge this to the session folder(EEG_data\T-001\Session 1).

  process new participant:
  python MI\process_new_participant.py --participant T-001 --base-path C:\Users\yuzeb\EEG_data --sessions Session 1

  Training:
  python MI\classifiers\subject_specific\train_subject_classifier.py --participant T-001


  Real time deploy:
  python simulation/eeg_model_integration_v2.py --participant T-001