# This pipeline should perform the following steps:
# 1. Fetch all available radar and ecg data from a data folder and create train/test split
# 2. Optimize (Preprocessing + Feature Generation + Model Training) to arrive at optimal Parameters for each step
#   -> optimal parameters are those that yield the highest results in a predefined evaluation metric
#       -> this metric could be:
#           1. The mean absolute error of the predicted R-peaks (max of gaussians) and the real ones
#           2. The complete gaussian label signal subtracted from the predicted one
# 3. Result is a trained model that can be run on new sequential data (even recorded live) that predicts R-peaks (i.e. first heart sound) from
#   radar data

# Pseudo-Pipeline:
# 1. Load radar, Load ECG
# 2. Preprocessing: Highpass-Filter raw radar (1000Hz)
# 3. Feature Generation: Compute+Lowpass-filter (15Hz) Radar Power, Compute Radar Angle, Compute 15-80Hz Radar Hilbert-Enevelope
#   -> Decimate all three & rad_i & rad_q by factor 20 (-> new sfreq 100Hz)
#   -> 5 features => input vector of shape (num_samples, TIMESTEPS, 5)
# 4. Label Generation: Find R-Peaks in reference ECG (100Hz) and convolve resulting spike train with Gaussians of fixed length



from sklearn.model_selection import GroupKFold, ParameterGrid
from tpcp import Dataset, Algorithm, OptimizableParameter, OptimizablePipeline, cf
from tpcp.optimize import GridSearch, GridSearchCV

from empkins_micro.emrad.models.biLSTM import *
from empkins_micro.emrad.preprocessing.pre_processing_algorithms import *
from empkins_micro.emrad.feature_extraction.feature_generation_algorithms import *
from empkins_micro.emrad.label_generation.label_generation_algorithms import *

class PreProcessor(Algorithm):
    """Class preprocessing the radar to arrive at the heart sound envelope

    Result: self.processed_radar_
    """
    
    _action_methods = "pre_process"

    # Input Parameters
    highpass_filter: ButterHighpassFilter
    bandpass_filter: ButterBandpassFilter
    envelope_algo: ComputeEnvelopeSignal
    decimation_algo: ComputeDecimateSignal

    # Results
    processed_radar_: pd.Series

    def __init__(
        self,
        highpass_filter: ButterHighpassFilter = cf(ButterHighpassFilter()),
        bandpass_filter: ButterBandpassFilter = cf(ButterBandpassFilter()),
        envelope_algo: ComputeEnvelopeSignal = cf(ComputeEnvelopeSignal()),
        decimation_algo: ComputeDecimateSignal = cf(ComputeDecimateSignal(downsampling_factor=10))
    ):
        self.highpass_filter = highpass_filter
        self.bandpass_filter = bandpass_filter
        self.envelope_algo = envelope_algo
        self.decimation_algo = decimation_algo

    @make_action_safe
    def pre_process(self, raw_radar: pd.DataFrame):
        """Preprocess radar data that has been synced and sr aligned (to 1000Hz) already

        Args:
            raw_radar (pd.DataFrame): synced and sr aligned radar of one antenna
        
        Returns:
            self
        """

        highpass_filter_clone = self.highpass_filter.clone()
        bandpass_filter_clone = self.bandpass_filter.clone()
        envelope_algo_clone = self.envelope_algo.clone()
        decimation_algo_clone = self.decimation_algo.clone()

        # Get rid of the freq=0 offset
        highpassed_radi = highpass_filter_clone.filter(raw_radar['I'], sample_frequency_hz=1000)
        highpassed_radq = highpass_filter_clone.filter(raw_radar['Q'], sample_frequency_hz=1000)

        # Compute the radar power from I and Q
        rad_power = np.sqrt(np.square(highpassed_radi.filtered_signal_)+np.square(highpassed_radq.filtered_signal_))

        # Extract heart sound band and compute the hilbert envelope
        heart_sound_radar = bandpass_filter_clone.filter(rad_power, 1000)
        heart_sound_radar_envelope = envelope_algo_clone.compute(heart_sound_radar.filtered_signal_)

        # Downsample to 100 Hz
        heart_sound_radar_envelope = decimation_algo_clone.compute(heart_sound_radar_envelope.envelope_signal_).downsampled_signal_

        # Normalize and return the envelope
        mean = heart_sound_radar_envelope.mean(axis=0)
        std = heart_sound_radar_envelope.std(axis=0)
        self.processed_radar_ = (heart_sound_radar_envelope - mean) / std

        return self
    
    
class InputAndLabelGenerator(Algorithm):
    """Class generating the Input and Label matrices for the BiLSTM model for sitting participants

    Results: 
        self.input_data
        self.input_labels
    """
    
    _action_methods = ("generate_training_input_sitting", "generate_training_labels_sitting")

    # PreProcessing
    pre_processor: PreProcessor

    # Label Generation
    label_decimation_algo: ComputeDecimateSignal
    peak_gaussian_algo: ComputeEcgPeakGaussians

    # Input & Label parameters
    timesteps: int
    step_size: int
    
    # Results
    input_data_: np.ndarray
    input_labels_: np.ndarray

    def __init__(
        self,
        pre_processor: PreProcessor = cf(PreProcessor()),
        label_decimation_algo: ComputeDecimateSignal = cf(ComputeDecimateSignal(downsampling_factor=10)),
        peak_gaussian_algo: ComputeEcgPeakGaussians = cf(ComputeEcgPeakGaussians()),
        timesteps: int = 400,
        step_size: int = 20
    ):
        self.pre_processor = pre_processor
        self.label_decimation_algo = label_decimation_algo
        self.peak_gaussian_algo = peak_gaussian_algo
        self.timesteps = timesteps
        self.step_size = step_size

    @make_action_safe
    def generate_training_input_sitting(self, dataset: MicroBaseDataset):
        """Method for generating training input

        Args:
            dataset (MicroBaseDataset): The dataset to compute input data for.

        Returns:
            self
        """
        res = []

        pre_processor_clone = self.pre_processor.clone()

        # loop over dataset
        for group in dataset:

            # fetch the radar data
            data_dict = group.emrad_biopac_synced_and_sr_aligned

            # preprocess the radar data
            heart_sound_band_envelope_rad1 = pre_processor_clone.pre_process(data_dict['rad1_aligned__resampled_']).processed_radar_
            heart_sound_band_envelope_rad2 = pre_processor_clone.pre_process(data_dict['rad2_aligned__resampled_']).processed_radar_

            # generate input samples
            for i in range(0, len(heart_sound_band_envelope_rad1) - self.timesteps, self.step_size):
                rad1 = heart_sound_band_envelope_rad1[i:(i + self.timesteps)]
                rad2 = heart_sound_band_envelope_rad2[i:(i + self.timesteps)]
                rad1 = np.expand_dims(rad1, axis=(1))
                rad2 = np.expand_dims(rad2, axis=(1))
                combined_rad = np.concatenate((rad1, rad2), axis=1)
                res.append(combined_rad)
        
        # safe input samples
        self.input_data_ = np.array(res)
        return self
    
    @make_action_safe
    def generate_training_labels_sitting(self, dataset: MicroBaseDataset):
        """Method for generating the labels for the BiLSTM training phase

        Args:
            dataset (MicroBaseDataset): The dataset to generate labels for.

        Returns:
            self
        """
        res = []

        label_decimation_algo_clone = self.label_decimation_algo.clone()
        peak_gaussian_algo_clone = self.peak_gaussian_algo.clone()

        # loop over dataset
        for group in dataset:
            # get the synced biopac data
            data_dict = group.emrad_biopac_synced_and_sr_aligned

            # downsample the ecg data
            downsampled_ecg = label_decimation_algo_clone.compute(data_dict['Biopac_aligned__resampled_']['ecg']).downsampled_signal_

            # compute the peak gaussians which will be labels
            peak_gaussian_signal = peak_gaussian_algo_clone.compute(downsampled_ecg, 1000 / self.label_decimation_algo.downsampling_factor).peak_gaussians_
           
            # normalize the label data
            mean = np.mean(peak_gaussian_signal)
            std = np.std(peak_gaussian_signal)
            peak_gaussian_signal = (peak_gaussian_signal - mean) / std

            # generate labels
            for i in range(0, len(peak_gaussian_signal) - self.timesteps, self.step_size):
                next_sample = peak_gaussian_signal[i:(i + self.timesteps)]
                res.append(next_sample)

        # safe labels
        self.input_labels_ = np.array(res)
        return self


class BiLstmPipeline(OptimizablePipeline):

    feature_extractor: InputAndLabelGenerator
    lstm: BiLSTM
    lstm___model: OptimizableParameter

    result_ = np.ndarray

    def __init__(
        self,
        feature_extractor: InputAndLabelGenerator = cf(InputAndLabelGenerator()),
        lstm: BiLSTM = cf(BiLSTM())
    ):
        self.feature_extractor = feature_extractor
        self.lstm = lstm

    def self_optimize(self, dataset: MicroBaseDataset):
        # Get data from dataset
        feature_extractor_clone = self.feature_extractor.clone()
        input_data = feature_extractor_clone.generate_training_input_sitting(dataset).input_data_
        input_labels = feature_extractor_clone.generate_training_labels_sitting(dataset).input_labels_
        # Create and fit model
        self.lstm = self.lstm.clone()
        self.lstm.self_optimize(input_data, input_labels)
        return self
        
    def run(self, datapoint: MicroBaseDataset):
        # Get data from dataset
        input_data = self.feature_extractor.generate_training_input_sitting(datapoint).input_data_

        # model predict
        lstm_copy = self.lstm.clone()
        lstm_copy = lstm_copy.predict(input_data)
        self.result_ = lstm_copy.predictions_

        return self

# TODO: Implement one scorer per use case
# Use cases are:
# 1. Good heart rate extimation over the long run
# 2. Good accuracy important for HRV computation

# def scoring(pipeline, datapoint):
#     pipeline.run(datapoint)

#     result = pipeline.result_

#     labels = datapoint.labels

#     erorres = f1_score(results, labels)

#     return {"f1_score": error, "per_rpeak": CustomAgg((results, labels)), "raw": NoAgg(results)}


  
#pickle.dump
#pickle.load



    