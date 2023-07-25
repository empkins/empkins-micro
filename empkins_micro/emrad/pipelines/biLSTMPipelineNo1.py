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
from tpcp import Dataset, OptimizableParameter, OptimizablePipeline
from tpcp.optimize import GridSearch, GridSearchCV


class MyPipeline(OptimizablePipeline):

    lstm___model: OptimizableParameter

    def __init__(self, lstm, pre_process, cutoff: float):
        self.lstm = lstm
        self.pre_process = pre_process

    def self_optimize(self, dataset):
        # Get data from dataset
        data = [d.data for d in dataset]
        labels = ...
        prpcessed_data = self._preprocess_data(data)
        # Create and fit model
        self.lstm = self.lstm.clone()
        self.lstm.fit()
        return self
        
    def run(self, datapoint):
        # Get data from dataset
        data = datapoint.data
        prepcessed_data = self._preprocess_data(data)

        # model predict
        self.result_ = model_prediction

        pass 
    def _preprocess_data(self, data):

        PreProcessAlgo2(self.cutoff)
        pass

lstm = BiLSTM()
test = MyPipeline(lstm=lstm, pre_process=PreProcessor())
test2 =  MyPipeline(lstm=lstm, pre_process=PreProcessor())


dataset = Dataset()

test.self_optimize(dataset[0:10])
results = [test.run(d).results_ for d in dataset[10:20]]

para_grid = ParameterGrid({"lstm__input_shape": , "pre_processor__para":..., "cutoff"})

def scoring(pipeline, datapoint):
    pipeline.safe_run(datapoint)

    results = pipeline.results_

    labels = datapoint.labels

    erorres = f1_score(results, labels)

    return {"f1_score": error, "per_rpeak": CustomAgg((results, labels)), "raw": NoAgg(results)}

gs = GridSearchCV(test, parameter_grid=para_grid, scoring=scoring, return_optimized="f1_score", cv=GroupKFold())
gs.optimize(dataset, groups=dataset.create_group_labels("participant"))
gs.optimized_pipeline_
pd.DataFrame(gs.cv_results_)