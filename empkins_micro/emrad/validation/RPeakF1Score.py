from tpcp import Algorithm, make_action_safe
import numpy as np
from scipy import interpolate
from scipy.signal import find_peaks

class RPeakF1Score(Algorithm):
    """Algorithm object applying the scipy.signal find_peaks() function to find R-Peaks in the predicted R-Peak signal from the biLSTMPipelineNo1
    and the ground truth data. Afterwards it calculates the precision (true positives out of retrieved positives) and recall (true positives out 
    of true positives and false negatives) and from that the F1-score of the predicted R-Peaks.        
    """

    _action_methods = "compute"

    # INPUT PARAMETERS
    max_deviation_ms: int
    max_heart_rate: int
    sampling_rate: float

    # Results
    f1_score_: np.ndarray

    def __init__(
        self,
        max_deviation_ms: int = 50,
        max_heart_rate: int = 180,
        sampling_rate: float = 100
    ):
        self.max_deviation_ms = max_deviation_ms
        self.max_heart_rate = max_heart_rate
        self.sampling_rate = sampling_rate

        
    @make_action_safe
    def compute(self, predicted_r_peak_signal: np.ndarray, ground_truth_r_peak_signal: np.ndarray):
        """Accepts a R peak prediciton signal and R peak ground truth and computes the F1 score of the prediciton.

        Args:
            predicted_r_peak_signal (np.ndarray)
            ground_truth_r_peak_signal (np.ndarray)

        Returns:
            self
        """

        max_deviation_samples = self.sampling_rate * self.max_deviation_ms / 1000

        minimal_distance_between_peaks =  int(1 / (self.max_heart_rate / 60) * self.sampling_rate)
        
        pred_peaks, _ = find_peaks(predicted_r_peak_signal, distance=minimal_distance_between_peaks, prominence=0.15)
        gt_peaks, _ = find_peaks(ground_truth_r_peak_signal, distance=minimal_distance_between_peaks, prominence=0.15)

        true_positives = 0
        
        next_gt = 0

        next_pred = 0

        while next_gt < len(gt_peaks):

            current_distance = abs(gt_peaks[next_gt] - pred_peaks[next_pred])

            if next_pred + 1 > len(pred_peaks) - 1:

                if current_distance < max_deviation_samples:

                    true_positives += 1
                    break
            
            next_distance =  abs(gt_peaks[next_gt] - pred_peaks[next_pred + 1])

            while next_distance < current_distance:

                current_distance = next_distance

                next_pred += 1

                if next_pred + 1 > len(pred_peaks) - 1:

                    if current_distance < max_deviation_samples:

                        true_positives += 1
                        break
                
                next_distance = abs(gt_peaks[next_gt] - pred_peaks[next_pred + 1])
            
            if current_distance < max_deviation_samples:

                true_positives += 1

                # do not count a peak twice as true positive
                next_pred += 1
            
            if next_pred > len(pred_peaks) - 1:
                break

            next_gt += 1
        
        precision = true_positives / len(pred_peaks)

        recall = true_positives / len(gt_peaks)

        self.f1_score_ = 2 * ((precision * recall) / (precision + recall))
        
        return self
    