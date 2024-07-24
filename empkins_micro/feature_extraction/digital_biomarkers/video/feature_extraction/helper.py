import numpy as np

def calculate_mean_distance(df, region_range):
    x_avg = df[[f'X{i}' for i in region_range]].mean(axis=1)
    y_avg = df[[f'Y{i}' for i in region_range]].mean(axis=1)
    z_avg = df[[f'Z{i}' for i in region_range]].mean(axis=1)
    
    distance = np.sqrt(x_avg.diff()**2 + y_avg.diff()**2 + z_avg.diff()**2)
    return distance.mean()
    
def calculate_range_of_motion(df, region_range):
    x_avg = df[[f'X{i}' for i in region_range]].mean(axis=1)
    y_avg = df[[f'Y{i}' for i in region_range]].mean(axis=1)
    z_avg = df[[f'Z{i}' for i in region_range]].mean(axis=1)
    
    rom_x = x_avg.max() - x_avg.min()
    rom_y = y_avg.max() - y_avg.min()
    rom_z = z_avg.max() - z_avg.min()
    
    return [(rom_x, rom_y, rom_z)]

def calculate_velocity(df, region_range, time=1, correction=None):
    # This is a rough approximation using positional data
    x_avg = df[[f'X{i}' for i in region_range]].mean(axis=1)
    y_avg = df[[f'Y{i}' for i in region_range]].mean(axis=1)
    z_avg = df[[f'Z{i}' for i in region_range]].mean(axis=1)
    
    if correction is not None:
        x_corr = df[[f'X{i}' for i in correction]].mean(axis=1)
        y_corr = df[[f'Y{i}' for i in correction]].mean(axis=1)
        z_corr = df[[f'Z{i}' for i in correction]].mean(axis=1)
        
        x_avg = x_avg - x_corr
        y_avg = y_avg - y_corr
        z_avg = z_avg - z_corr
        
    dx = x_avg.diff()
    dy = y_avg.diff()
    dz = z_avg.diff()
    
    # Approximate angular velocity as rate of change of direction
    velocity = np.sqrt(dx**2 + dy**2 + dz**2) / time
    
    return velocity

def calculate_std_dev_movement(df, region_range):
    x_avg = df[[f'X{i}' for i in region_range]].mean(axis=1)
    y_avg = df[[f'Y{i}' for i in region_range]].mean(axis=1)
    z_avg = df[[f'Z{i}' for i in region_range]].mean(axis=1)
    
    std_dev_x = x_avg.std()
    std_dev_y = y_avg.std()
    std_dev_z = z_avg.std()
    
    return [(std_dev_x, std_dev_y, std_dev_z)]

def visibility(pose, region_range, threshold=0.8):
    region = pose[[f'visibility{i}' for i in region_range]]   
    return (region > threshold).sum().mean() / len(pose)

def static_periods(velocity, frameRate, threshold = 0.3):
    # static if total std signal is below threshold for 1 s 
    step = int(frameRate // 2)
    count = 0
    windows = 0
    for i in range(step, len(velocity) - step, step):
        std = velocity.iloc[i-step:i+step+1].std()
        windows +=1
        if std < velocity.max() * threshold:
            count +=1
    return count / windows
    

def below_threshold(velocity, threshold=0.1):
    # 2 norm < 0.1 * max(x)
    return (velocity < (velocity.max() * threshold)).sum()