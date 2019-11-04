import numpy as np
np.random.seed(42)

if __name__ == '__main__':
    # get times at which observations begin:
    start_times = []
    start_times.append([n+np.random.normal(0.05, 0.01) for n in range(60)])
    start_times.append([n+np.random.normal(0.15, 0.01) for n in range(60)])
    start_times.append([n+np.random.normal(0.25, 0.01) for n in range(60)])
    start_times = np.array(start_times)
    start_times = np.append(start_times[start_times < 29.5], 
                            start_times[start_times > 30.5]) # remove asteroseismic night
                            
    # populate full series of timestamps at which to calculate RV:
    t = np.array([])
    step = 1./60/24 # 1 minute in days
    for tt in start_times:
        t = np.append(t, np.arange(tt,tt+30*step,step, dtype=np.float64)) # normal observing nights
    t = np.append(t, np.arange(30.0,30.3,step, dtype=np.float64)) # asteroseismic night
    t.sort()
    
    # save:
    np.savetxt('simulated_times.txt', t, header='Date (days)')