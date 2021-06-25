import mne
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA
import sys
sys.path.append('/neurospin/meg/meg_tmp/Geom_Seq_Fosca_2017/GeomSeq_New/')
from GeomSeq_analyses import config
import os.path as op
import pandas as pd
import numpy as np
import glob
import os

def trigger_delay():
    trigger_delay = {'sequences': 50, 'pairs': 50, 'ling_vis': 54}
    return trigger_delay


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def compute_epochs(subject, tmin, tmax, decim=1, reject = None, baseline=None, suffix="",block_type = 'primitives', filter=None):
    """
    The function epochs the raw data and appends the corresponding metadata dataframe to it.
    :param subject: subject id
    :param tmin: minimal time for epoching
    :param tmax: maximal time for epoching
    :param decim: params from the mne.epoching func
    :param reject: params from the mne.epoching func
    :param baseline: params from the mne.epoching func
    :param suffix: allows to save with another name if you use special epoching params
    :param block_type: set it to 'primitives', 'sequences' or 'localizer' depending on which type of run you want to epoch
    :param filter: pandas query to select just a subpart of the epochs to build the epochs
    :return: True
    """

    # ---- preprocessed data path ----
    meg_subject_dir = config.data_path+subject+'/processed_data_ica/'
    subjects_runs = config.runs_dict[subject]
    raw_list = []

    # ---- epoching for the different types of blocks ----
    if block_type=='primitives':
        key_word = "pairs"
        n_epochs = 768
    elif block_type == 'sequences':
        key_word = 'seq'
        n_epochs = 1152
    elif block_type == 'localizer':
        key_word = 'loc'
        n_epochs = 800
    else:
        print('The argument block_type should be either primitives, sequences or localizer')

    for run in subjects_runs:
        # ========= we save one epoch object per run ======
        if key_word in run:
            extension = run + '_raw_sss'
            if key_word != 'loc':
                ii = int(run[-1])
            raw_fname_in = op.join(meg_subject_dir, config.base_fname.format(**locals()))
            raw = mne.io.read_raw_fif(raw_fname_in, preload=True)

            # ====== we remove the response events ===========
            events = mne.find_events(raw, stim_channel=config.stim_channel,
                                     consecutive=True,
                                     min_duration=config.min_event_duration,
                                     shortest_event=config.shortest_event, mask=1024 + 2048 + 4096 + 8192 + 16384 + 32768,mask_type='not_and')

            trigger_del = trigger_delay()
            # --- the trigger delay is the same for the primitives, the sequences and the localizer parts
            events[:, 0] = events[:, 0] + trigger_del['sequences']
            print('We have corrected for the trigger delay ')

            if len(events) != n_epochs:
                raise ValueError(" problematic subject %s that has for run %s %i events instead of %i"%(subject,run,len(events),n_epochs))

            # the metadata is computed depending of the
            if block_type=='primitives':
                metadata = extract_metadata_PAIRS(events,ii+1)
            elif block_type == "localizer":
                metadata = extract_metadata_LOCALIZER(events)
            else:
                metadata = extract_metadata_SEQUENCES(events, ii + 1)

            picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False, exclude=())
            epoch = mne.Epochs(raw, events, None, tmin,
                                                    tmax, proj=True,
                                                    picks=picks, decim=decim, reject=reject, baseline=baseline)

            epoch.metadata = metadata

            if filter is not None:
                epoch = epoch[filter]

            epoch_save = config.saving_path + '/epochs/' + subject + '/'
            create_folder(epoch_save)
            epoch_savename = epoch_save + run + "-epo.fif"
            if len(suffix)>0:
                eposave = epoch_save+suffix+'/'
                print(eposave)
                create_folder(eposave)
                epoch_savename = eposave + run  +"-epo.fif"

            epoch.save(epoch_savename)
            del epoch
            del raw

    return True

# ______________________________________________________________________________________________________________________
def extract_metadata_SEQUENCES(events,run_numb):


    sequence_ID = {10:'repeat',20:'alternate',30:'4diagonals', 40:'4segments', 50:'2crosses', 60:'2arcs',
                 70:'2squares', 80:'2rectangles', 90:'irregular', 100:'memory1', 110:'memory2', 120:'memory4'}

    primitive_ID = {'rotp1':10,'rotm1':20,'rotp2':30,'rotm2':40,'rotp3':50,'rotm3':60,
                    'sym_point':70,'H':80,'V':90,'A':100,'B':110,'control':120}

    primitive_df = sequences_primitives()
    struct_df = sequences_hierarchical_structure()

    # ======== general run related information ==============================
    run_number = ([run_numb]*1152)
    subrun_number = np.concatenate([[i]*96 for i in range(1,13)])
    position_in_subrun = np.concatenate([range(1,97) for i in range(1,13)])
    position_in_sequence = np.concatenate([[k for k in range(1,9)]*12 for i in range(1,13)])
    block_type = ['sequences']*1152

    # ======== stimulus related information ==============================
    position_on_screen = []
    violation = []
    sequence = []
    sequence_subtype = []
    position_pair = []
    violation_primitive = []
    # ======== primitive related information ==============================
    primitive = []
    primitive_level = []
    primitive_level1 = []
    primitive_level2 = []

    # ======== structure related information ==============================
    SequenceOrdinalPosition = []
    WithinComponentPosition = []
    ComponentBeginning = []
    ComponentEnd = []

    # ======= %%%%% we start the loop %%%%%%% =============
    for num in range(12):

        for k in range(96):
            eve = events[num*96+k,2]
            if eve>128:
                violation.append(1)
                eve = - eve + 255
                violation_primitive[-1]=1
                violation_primitive.append(1)
            else:
                violation.append(0)
                violation_primitive.append(0)

            pos_on_screen = eve%10
            seqID = eve - pos_on_screen
            which_seq = sequence_ID[seqID]
            position_on_screen.append(pos_on_screen)
            sequence.append(which_seq)
            if k<95:
                position_pair.append([int(pos_on_screen * 10 + events[num*96+k+1,2] % 10)])
            else:
                position_pair.append(['nan'])

        seq_subtype = identify_sequence_subtype(position_on_screen[-32:-24])

        print("========== THE SEQUENCE TYPE IS %s ============="%which_seq)
        print("========== THE SEQUENCE PRESENTED SEQUENCE IS =============")
        print(position_on_screen[-8:])
        print("========== THE SEQUENCE SUBTYPE IS %s ============="%seq_subtype)
        sequence_subtype.append([seq_subtype]*96)

        struct_seq = struct_df[struct_df['sequence']==which_seq]
        SequenceOrdinalPosition.append(struct_seq['sequence_item'].values.tolist()*12)
        WithinComponentPosition.append(struct_seq['WithinCompPosition'].values.tolist()*12)
        ComponentBeginning.append(struct_seq['CompBeginning'].values.tolist()*12)
        ComponentEnd.append(struct_seq['CompEnd'].values.tolist()*12)

        primitives_df = primitive_df[primitive_df['sequence_subtype']==seq_subtype]
        primitive.append(primitives_df['Primitives'].values.tolist()*12)
        primitive_level.append(primitives_df['PrimitivesHierarchy'].values.tolist() * 12)
        primitive_level1.append(primitives_df['Primitives_level1'].values.tolist() * 12)
        primitive_level2.append(primitives_df['Primitives_level2'].values.tolist() * 12)

    primitive = np.hstack(primitive)
    primitive_code = [primitive_ID[primitive[m]] if primitive[m]!='nan' else np.nan for m in range(len(primitive))]


    metadata = {'run_number':np.squeeze(np.asarray(run_number)),
                'subrun_number': np.squeeze(np.asarray(subrun_number)),
                'position_in_subrun': np.squeeze(np.asarray(position_in_subrun)),
                'position_in_sequence': np.squeeze(np.asarray(position_in_sequence)),
                'block_type': np.squeeze(np.asarray(block_type)),
                'position_on_screen': np.squeeze(np.asarray(position_on_screen)),
                'violation': np.squeeze(np.asarray(violation)),
                'violation_primitive': np.squeeze(np.asarray(violation_primitive)),
                'sequence': np.squeeze(np.asarray(sequence)),
                'sequence_subtype': np.squeeze(np.hstack(sequence_subtype)),
                'primitive': np.squeeze(np.asarray(primitive)),
                'primitive_level': np.squeeze(np.hstack(primitive_level)),
                'primitive_code': np.squeeze(np.asarray(primitive_code)),
                'SequenceOrdinalPosition': np.squeeze(np.hstack(SequenceOrdinalPosition)),
                'WithinComponentPosition': np.squeeze(np.hstack(WithinComponentPosition)),
                'ComponentBeginning': np.squeeze(np.hstack(ComponentBeginning)),
                'ComponentEnd': np.squeeze(np.hstack(ComponentEnd)),
                'position_pair':np.squeeze(np.hstack(position_pair)),
                'primitive_level1' :np.squeeze(np.hstack(primitive_level1)),
                'primitive_level2 ': np.squeeze(np.hstack(primitive_level2))
                }

    metadata_df = pd.DataFrame.from_dict(metadata)

    return metadata_df

# ______________________________________________________________________________________________________________________
def extract_metadata_PAIRS(events,run_numb):

    primitive_ID = {10:'rotp1',20:'rotm1',30:'rotp2', 40:'rotm2', 50:'rotp3', 60:'rotm3',
                 70:'sym_point', 80:'H', 90:'V', 100:'A', 110:'B', 120:'control'}

    first_or_second = np.concatenate([[1,2]*(32*12)])
    miniblock_number = np.concatenate([[i]*64 for i in range(1,13)])
    pair_number_in_miniblock = np.concatenate([[i]*2 for i in range(1,33)]*12)
    run_number = ([run_numb]*768)

    violation = []
    primitive = []
    rotation_or_symmetry = []
    position_on_screen = []
    position_pair = []
    primitive_code = []
    block_type = ['primitives']*768

    for k in range(len(events)):

        eve = events[k,2]
        if eve>128:
            violation[-1] = 1
            violation.append(1)
            eve = - eve + 255
        else:
            violation.append(0)

        pos_on_screen = eve%10
        prim_key = eve - pos_on_screen
        primitive_code.append(prim_key)
        if prim_key in [10,20,30,40,50,60]:
            rotation_or_symmetry.append('rotation')
        elif prim_key in [80,90,100,110]:
            rotation_or_symmetry.append('symmetry')
        else:
            rotation_or_symmetry.append('other')

        position_on_screen.append(pos_on_screen)
        primitive.append(primitive_ID[prim_key])

        if k%2 ==0:
            if events[k+1,2] < 128:
                position_pair.append([pos_on_screen*10+events[k+1,2]%10])
            else:
                position_pair.append([pos_on_screen * 10 + (- events[k+1,2] + 255) % 10])
        else:
            position_pair.append([position_on_screen[-2]*10+pos_on_screen])


    metadata = {'first_or_second':np.squeeze(np.asarray(first_or_second)),
                'miniblock_number':np.squeeze(np.asarray(miniblock_number)),
                'pair_number_in_miniblock':np.asarray(pair_number_in_miniblock),
                'run_number':np.asarray(run_number),
                'violation':np.asarray(violation),
                'primitive':np.asarray(primitive),
                'rotation_or_symmetry':np.asarray(rotation_or_symmetry),
                'position_on_screen':np.asarray(position_on_screen),
                'position_pair':np.squeeze(np.asarray(position_pair)),
                'primitive_code':np.squeeze(np.asarray(primitive_code)),
                'block_type':np.squeeze(np.asarray(block_type))
                }

    metadata_df = pd.DataFrame.from_dict(metadata)

    return metadata_df



# ______________________________________________________________________________________________________________________
def extract_metadata_LOCALIZER(events):


    events, events_first_ID = change_trigger_localizer(events)

    miniblock_number = [1]*800
    run_number = [1]*800
    violation = []
    pos_on_screen = []
    block_type = ['localizer']*800

    for k in range(len(events)):
        eve = events[k,2]
        if eve==9:
            violation.append(1)
        else:
            violation.append(0)
        pos_on_screen.append(eve)

    metadata = {'position_on_screen':np.squeeze(np.asarray(pos_on_screen)),
                'miniblock_number':np.squeeze(np.asarray(miniblock_number)),
                'run_number':np.asarray(run_number),
                'violation':np.asarray(violation),
                'block_type':np.squeeze(np.asarray(block_type))
                }

    metadata_df = pd.DataFrame.from_dict(metadata)

    return metadata_df



# ______________________________________________________________________________________________________________________
def sequences_hierarchical_structure():

    """ From the sequence_ID, retreive all the information related to the sequence structure.
    """
    print("=============== In this version there is nothing coded for 2rectangles ===========================")
    print("========== Irregular_pilot corresponds to a sequence that was shown only for the macaque =========")

    WithinCompPosition = {'repeat':[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                          'alternate':[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                          '4diagonals':[1,2,1,2,1,2,1,2],
                          '4segments':[1,2,1,2,1,2,1,2],
                          '2crosses':[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                          '2arcs':[1,2,3,4,1,2,3,4],
                          '2squares':[1,2,3,4,1,2,3,4],
                          '2rectangles':[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                           'irregular':[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                          'memory1':[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                          'memory2':[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                          'memory4':[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                     'irregular_pilot': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]}

    CompBeginning = {'repeat': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                          'alternate': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                          '4diagonals': [1,0,1,0,1,0,1,0],
                          '4segments': [1,0,1,0,1,0,1,0],
                          '2crosses': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                          '2arcs': [1,0,0,0,1,0,0,0],
                          '2squares': [1,0,0,0,1,0,0,0],
                          '2rectangles': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                          'irregular': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                          'memory1': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                          'memory2': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                          'memory4': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                     'irregular_pilot': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]}

    CompEnd = {'repeat': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                          'alternate': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                          '4diagonals': [0,1,0,1,0,1,0,1],
                          '4segments': [0,1,0,1,0,1,0,1],
                          '2crosses': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                          '2arcs': [0,0,0,1,0,0,0,1],
                          '2squares': [0,0,0,1,0,0,0,1],
                          '2rectangles': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                          'irregular': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                          'memory1': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                          'memory2': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                          'memory4': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
               'irregular_pilot': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]}

    sequence = []
    sequence_item = []
    withincomposition = []
    compbeginning = []
    compend = []

    for seq in WithinCompPosition.keys():
        sequence.append([seq]*8)
        sequence_item.append([i for i in range(1,9)])
        withincomposition.append(WithinCompPosition[seq])
        compbeginning.append(CompBeginning[seq])
        compend.append(CompEnd[seq])

    struct_dic = {'sequence':np.hstack(sequence),'sequence_item':np.hstack(sequence_item),
                  'WithinCompPosition':np.hstack(withincomposition),'CompBeginning':np.hstack(compbeginning),'CompEnd':np.hstack(compend)}

    data_Frame = pd.DataFrame.from_dict(struct_dic)
    return data_Frame



# ______________________________________________________________________________________________________________________
def identify_sequence_subtype(seq_positions):
    """
    From the list of sequence positions, this function returns the sequence versions V1 or V2.
    We have to give as key value ev = (events_to_identify_sequence - events_to_identify_sequence[0])%8

    :return:
    """

    # ======== determine the corresponding string of positions ===========
    ev = (seq_positions - seq_positions[0])%8
    ev = (ev.tolist())
    init = seq_positions[0] % 10

    # ======= then consider the possible subcases =========================
    versions = [''] * 8
    if ev == [0, 7, 1, 6, 2, 5, 3, 4]:
        sub_seq = '4segmentsV1'
        versions = ['B','H','A','V','B','H','A','V']
    elif ev == [0, 1, 7, 2, 6, 3, 5, 4]:
        sub_seq = '4segmentsV2'
        versions = ['H','A','V','B','H','A','V','B']
    elif ev == [0, 1, 2, 3, 7, 6, 5, 4]:
        sub_seq = '2arcsV1'
        versions = ['B','H','A','V','B','H','A','V',]
    elif ev == [0, 7, 6, 5, 1, 2, 3, 4]:
        sub_seq = '2arcsV2'
        versions = ['H','A','V','B','H','A','V','B']
    elif ev == [0, 3, 4, 7, 2, 5, 6, 1]:
        sub_seq = '2rectanglesV1'
        versions = ['AB','VH','BA','HV','AB','VH','BA','HV']
    elif ev == [0,5,4,1,6,3,2,7]:
        sub_seq = '2rectanglesV2'
        versions = ['VH','BA','HV','AB','VH','BA','HV','AB']
    elif ev == [0,1,2,3,4,5,6,7]:
        sub_seq = 'repeatV1'
    elif ev == [0,7,6,5,4,3,2,1]:
        sub_seq = 'repeatV2'
    elif ev == [0,3,2,5,4,7,6,1]:
        sub_seq = 'alternateV1'
    elif ev == [0,5,6,3,4,1,2,7]:
        sub_seq = 'alternateV2'
    elif ev == [0,4,1,5,2,6,3,7]:
        sub_seq = '4diagonalsV1'
    elif ev == [0,4,7,3,6,2,5,1]:
        sub_seq = '4diagonalsV2'
    elif ev == [0,4,3,7,6,2,1,5]:
        sub_seq = '2crossesV1'
    elif ev == [0,4,5,1,2,6,7,3]:
        sub_seq = '2crossesV2'
    elif ev == [0,2,4,6,1,3,5,7]:
        sub_seq = '2squaresV1'
    elif ev == [0,6,4,2,7,5,3,1]:
        sub_seq = '2squaresV2'
    else:
        sub_seq = 'other'
    sub_seq += versions[init]

    return sub_seq
# ______________________________________________________________________________________________________________________
def change_trigger_localizer(events):
    events_loc_mod = events.copy()
    events_first_ID = {'pos0': 0, 'pos1': 1, 'pos2': 2, 'pos3': 3, 'pos4': 4, 'pos5': 5, 'pos6': 6, 'pos7': 7}
    for i in range(len(events)):
        if events_loc_mod[i][2]>129:
            events_loc_mod[i][2]=events_loc_mod[i][2]-130
        else:
            events_loc_mod[i][2]= 9
    return events_loc_mod, events_first_ID
# ______________________________________________________________________________________________________________________
def sequences_primitives():

    """
    Things related to the primitives within a sequence. At which level they are applied etc.
    :return:
    """

    primitives = {'repeatV1':['rotp1','rotp1','rotp1','rotp1','rotp1','rotp1','rotp1','rotp1'],
                  'repeatV2':['rotm1','rotm1','rotm1','rotm1','rotm1','rotm1','rotm1','rotm1'],
                  'alternateV1':['rotp3','rotm1','rotp3','rotm1','rotp3','rotm1','rotp3','rotm1'],
                  'alternateV2':['rotm3','rotp1','rotm3','rotp1','rotm3','rotp1','rotm3','rotp1'],
                  '4diagonalsV1':['sym_point','rotp1','sym_point','rotp1','sym_point','rotp1','sym_point',np.nan],
                  '4diagonalsV2':['sym_point','rotm1','sym_point','rotm1','sym_point','rotm1','sym_point',np.nan],
                  '4segmentsV1A':['A','rotp1','A','rotp1','A','rotp1','A',np.nan],
                  '4segmentsV1B':['B','rotp1','B','rotp1','B','rotp1','B',np.nan],
                  '4segmentsV1H': ['H','rotp1','H','rotp1','H','rotp1','H',np.nan],
                  '4segmentsV1V': ['V','rotp1','V','rotp1','V','rotp1','V',np.nan],
                  '4segmentsV2A': ['A', 'rotm1', 'A', 'rotm1', 'A', 'rotm1', 'A', np.nan],
                  '4segmentsV2B': ['B', 'rotm1', 'B', 'rotm1', 'B', 'rotm1', 'B', np.nan],
                  '4segmentsV2H': ['H', 'rotm1', 'H', 'rotm1', 'H', 'rotm1', 'H',np.nan],
                  '4segmentsV2V': ['V', 'rotm1', 'V', 'rotm1', 'V', 'rotm1', 'V', np.nan],
                  '2crossesV1':['sym_point','rotm1','sym_point','rotm1','sym_point','rotm1','sym_point',np.nan],
                  '2crossesV2':['sym_point','rotp1','sym_point','rotp1','sym_point','rotp1','sym_point',np.nan],
                  '2arcsV1A':['rotp1','rotp1','rotp1','A','rotm1','rotm1','rotm1','A'],
                  '2arcsV1B':['rotp1','rotp1','rotp1','B','rotm1','rotm1','rotm1','B'],
                  '2arcsV1H':['rotp1','rotp1','rotp1','H','rotm1','rotm1','rotm1','H'],
                  '2arcsV1V':['rotp1','rotp1','rotp1','V','rotm1','rotm1','rotm1','V'],
                  '2arcsV2A':['rotm1','rotm1','rotm1','A','rotp1','rotp1','rotp1','A'],
                  '2arcsV2B':['rotm1','rotm1','rotm1','B','rotp1','rotp1','rotp1','B'],
                  '2arcsV2H':['rotm1','rotm1','rotm1','H','rotp1','rotp1','rotp1','H'],
                  '2arcsV2V':['rotm1','rotm1','rotm1','V','rotp1','rotp1','rotp1','V'],
                  '2squaresV1':['rotp2','rotp2','rotp2','rotp1','rotp2','rotp2','rotp2',np.nan],
                  '2squaresV2':['rotm2','rotm2','rotm2','rotm1','rotm2','rotm2','rotm2',np.nan],
                  '2rectanglesV1HV': ['H', 'sym_point', 'H', 'rotp2', 'V', 'sym_point', 'V', np.nan],
                  '2rectanglesV1AB': ['A', 'sym_point', 'A', 'rotp2', 'B', 'sym_point', 'B', np.nan],
                  '2rectanglesV1VH': ['V', 'sym_point', 'V', 'rotp2', 'H', 'sym_point', 'H', np.nan],
                  '2rectanglesV1BA': ['B', 'sym_point', 'B', 'rotp2', 'A', 'sym_point', 'A', np.nan],
                  '2rectanglesV2HV': ['H', 'sym_point', 'H', 'rotm2', 'V', 'sym_point', 'V', np.nan],
                  '2rectanglesV2AB': ['A', 'sym_point', 'A', 'rotm2', 'B', 'sym_point', 'B', np.nan],
                  '2rectanglesV2VH': ['V', 'sym_point', 'V', 'rotm2', 'H', 'sym_point', 'H', np.nan],
                  '2rectanglesV2BA': ['B', 'sym_point', 'B', 'rotm2', 'A', 'sym_point', 'A', np.nan],
                  'other':[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]}

    primitives_hierarchical_level ={'repeatV1': [0,0,0,0,0,0,0,0],
     'repeatV2': [0,0,0,0,0,0,0,0],
     'alternateV1': [0,0,0,0,0,0,0,0],
     'alternateV2': [0,0,0,0,0,0,0,0],
     '4diagonalsV1': [0,1,0,1,0,1,0,1],
     '4diagonalsV2': [0,1,0,1,0,1,0,1],
     '4segmentsV1A': [0,1,0,1,0,1,0,1],
     '4segmentsV1B': [0,1,0,1,0,1,0,1],
     '4segmentsV1H': [0,1,0,1,0,1,0,1],
     '4segmentsV1V': [0,1,0,1,0,1,0,1],
     '4segmentsV2A': [0,1,0,1,0,1,0,1],
     '4segmentsV2B': [0,1,0,1,0,1,0,1],
     '4segmentsV2H': [0,1,0,1,0,1,0,1],
     '4segmentsV2V': [0,1,0,1,0,1,0,1],
     '2crossesV1': [0,0,0,0,0,0,0,0],
     '2crossesV2': [0,0,0,0,0,0,0,0],
     '2arcsV1A': [0,0,0,1,0,0,0,1],
     '2arcsV1B': [0,0,0,1,0,0,0,1],
     '2arcsV1H': [0,0,0,1,0,0,0,1],
     '2arcsV1V': [0,0,0,1,0,0,0,1],
     '2arcsV2A': [0,0,0,1,0,0,0,1],
     '2arcsV2B': [0,0,0,1,0,0,0,1],
     '2arcsV2H': [0,0,0,1,0,0,0,1],
     '2arcsV2V': [0,0,0,1,0,0,0,1],
     '2squaresV1': [0,0,0,1,0,0,0,1],
     '2squaresV2': [0,0,0,1,0,0,0,1],
     '2rectanglesV1HV':[0,1,0,2,0,1,0,2],
     '2rectanglesV1AB':[0,1,0,2,0,1,0,2],
     '2rectanglesV1VH':[0,1,0,2,0,1,0,2],
     '2rectanglesV1BA':[0,1,0,2,0,1,0,2],
     '2rectanglesV2HV':[0,1,0,2,0,1,0,2],
     '2rectanglesV2AB':[0,1,0,2,0,1,0,2],
     '2rectanglesV2VH':[0,1,0,2,0,1,0,2],
     '2rectanglesV2BA':[0,1,0,2,0,1,0,2],
     'other': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]}

    primitives_level1 = {}
    primitives_level2 = {}

    for key in primitives:
        primitives_level1[key] = [np.nan]*8
        primitives_level2[key] = [np.nan]*8
        for i in range(8):
            if primitives_hierarchical_level[key][i]==0:
                primitives_level1[key][i]=primitives[key][i]
            else:
                primitives_level2[key][i] = primitives[key][i]

    sequence_subtype = []
    sequence_item = []
    Primitives = []
    Primitives_hierarchy = []
    Primitives_level1 = []
    Primitives_level2 = []

    for seq in primitives.keys():
        sequence_subtype.append([seq]*8)
        sequence_item.append([i for i in range(1,9)])
        Primitives.append(primitives[seq])
        Primitives_hierarchy.append(primitives_hierarchical_level[seq])
        Primitives_level1.append(primitives_level1[seq])
        Primitives_level2.append(primitives_level2[seq])


    struct_dic = {'sequence_subtype':np.hstack(sequence_subtype),'sequence_item':np.hstack(sequence_item),
                  'Primitives':np.hstack(Primitives),'PrimitivesHierarchy':np.hstack(Primitives_hierarchy),
                  'Primitives_level1':np.hstack(Primitives_level1),'Primitives_level2':np.hstack(Primitives_level2)}

    data_Frame = pd.DataFrame.from_dict(struct_dic)
    return data_Frame


# ------------------------------------------- ------------------------------------------- ------------------------------
def load_and_concatenate_epochs(subject, suffix="",folder_suffix="", no_rsa=True, filter='pairs'):

    """
    Function that loads all the files corresponding to one epoch type and that concatenates them into a single epoch object.
    :param subject:
    :param suffix:
    :param no_rsa:
    :param filter:
    :return:
    """
    epochs_names = glob.glob(config.saving_path + '/epochs/' + subject +'/'+folder_suffix+ '/*.fif')
    if len(suffix)>0:
        epochs_names = glob.glob(config.saving_path + '/epochs/' + subject +'/'+folder_suffix+ '/'+ suffix+'/*')
    print(epochs_names)

    epochs = []
    for epo in epochs_names:
        if no_rsa:
            if 'rsa' not in epo:
                if filter in epo:
                    epoch = mne.read_epochs(epo)
                    epochs.append(epoch)
        else:
            if filter in epo:
                epoch = mne.read_epochs(epo)
                epochs.append(epoch)

    epochs = mne.concatenate_epochs(epochs)

    return epochs

# ______________________________________________________________________________________________________________________
def sliding_window(epoch,sliding_window_size=25, sliding_window_step=1,
                                             sliding_window_min_size=None):
    """
    This function outputs an epoch object that was computed from a sliding window on the data
    :param epoch:
    :param sliding_window_size: Window size (number of data points) over which the data is averaged
    :param sliding_window_step: Step in number of data points. 4 corresponds to 10 ms.
    :param sliding_window_min_size: The last window minimal size.
    :return:
    """

    from umne import transformers

    xformer = transformers.SlidingWindow(window_size=sliding_window_size, step=sliding_window_step,
                                         min_window_size=sliding_window_min_size)

    n_time_points = epoch._data.shape[2]
    window_start = np.array(range(0, n_time_points - sliding_window_size + 1, sliding_window_step))
    window_end = window_start + sliding_window_size

    window_end[-1] = min(window_end[-1], n_time_points)  # make sure that the last window doesn't exceed the input size

    intermediate_times = [int((window_start[i] + window_end[i]) / 2) for i in range(len(window_start))]
    times = epoch.times[intermediate_times]

    epoch2 = mne.EpochsArray(xformer.fit_transform(epoch._data),epoch.info)
    epoch2._set_times(times)
    epoch2.metadata = epoch.metadata

    return epoch2
