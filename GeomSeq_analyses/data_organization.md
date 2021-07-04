## Scripts, data and results folders

Put in the config.py the paths where:
* **the raw and preprocessed data is**: data_path
* **you want to save the results**: results_path

```
└── Project_folder
    ├── data
    │   ├── epochs
    │   │   ├── sub01
    │   │   │   │   ├── localizer-epo.fif
    │   │   │   │   ├── the other epochs
    │   │   │   ├── rsa
    │   │   │   │   
    │   ├── processed_data_ica
    │   ├── rsa
    ├── figures
    ├── results
    │   ├── decoding
    │   │   ├── 11primitives
    │   │   ├── rotationVSsymmetry
    │   │   ├── stimulus
    │   │   ├── decode_ordinal_position_fullblock
    │   │   ├── ordinal_code_full_seq
    │   │   ├── ordipos_GAT
```
## *0_Create_epochs.py*

The preprocessed data should be in:

    meg_subject_dir = config.data_path + subject + '/processed_data_ica/'
    
and the epochs will be saved in:

```
epoch_save = config.saving_path + '/epochs/' + subject + '/'
```

where *subject* is the participant's identifier.
There are 3 main types of epochs corresponding to the 3 experimental parts: primitive, sequence and localizer parts.
##### The metadata fields for the epochs from the sequence part are:

* *run_number*: ranges from 1 to 4 (there are 4 runs in the task)
* *subrun_number*: ranges fro 1 to 12, it is the number of the repetition of the sequence (12 repetitions in total).
* *position_in_subrun*: index of the item in the 12*8 item presentations corresponding to a sequence in a run
* *position_in_sequence*: from 1 to 8, ordinal position in the sequence
* *block_type*: 'sequence'
* *position_on_screen*: from 1 to 8, the position of the dot flashed on the screen
* *violation*: 1 if the item was presented at a deviant location, else 0
* *violation_primitive*: 1 if the corresponding primitive operation was violated, else 0
* *sequence*: sequence name
* *sequence_subtype*: many sequences have several versions (depending e.g. of the global direction of rotation)
* *primitive*: the primitive that codes for the transition between the current item n and next one n+1
* *primitive_level*: 0, 1, or 2 depending on which level of the hierarchy it was applied  
* *primitive_code*: a numerical mapping of the primitive identity
* *SequenceOrdinalPosition*: from 1 to 8, ordinal position in the sequence
* *WithinComponentPosition*: For 2arcs, 2squares, 4diagonals and 4segments, it tells what is the ordinal position in the sequence
* *ComponentBeginning*: 1 if an item is opening a component and 0 otherwise.
* *ComponentEnd*: 1 if an item is ending a component and 0 otherwise.
* *position_pair*: 2-digits number representing the positions of the current dot on the screen and the next one
* *primitive_level1*: primitive operations at the first level of the embedding 
* *primitive_level2*: primitive operations at the second level of the embedding 

##### The metadata fields for the epochs from the pairs part are:

* *first_or_second*: if it is the first or the second item of the pair
* *miniblock_number*: 
* *run_number*: ranges from 1 to 4 (there are 4 runs in the task)
* *primitive*: the primitive that codes for the transition
* *violation*: 1 if the pair was violated, else 0
* *rotation_or_symmetry*: if the primitive belongs to the group of rotations or symmetries
* *position_on_screen*: from 1 to 8, the position of the dot flashed on the screen
* *position_pair*: 2-digits number representing the positions of the current dot on the screen and the next one
* *block_type*: 'primitives'
* *pair_number_in_miniblock*: from 1 to 32, the index of the pair in the mini-block
             
##### The metadata fields for the epochs from the localizer part are:

* *position_on_screen*: from 1 to 8, the position of the dot flashed on the screen
* *violation*: 1 if the pair was violated, else 0
* *block_type*: 'localizer'
* *run_number*: 1
* *miniblock_number*: 1
           
Once you have epoched your data in the right format, you are ready to run the other analyses.

