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
  - Trained MI model in MI/models/

  To train the model run:
  - python MI/process_new_participant.py (participant number)