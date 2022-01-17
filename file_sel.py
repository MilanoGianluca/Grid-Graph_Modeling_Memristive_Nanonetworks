from numpy import loadtxt, sort, concatenate

def file_sel(fdir, fname, SR, step):
    # RANDOM PARAMETERS FOR UNDEFINED FILES
    if fname == 'example_file_1.txt':
        data = loadtxt(fdir+fname)
        title = 'EXAMPLE FILE 1'
        start = 0
        stop = -1
        time = data[start:stop:step, 0]
        V = data[start:stop:step, 2]
        I_exp = data[start:stop:step, 1]
        
        if SR == 1:
            pos1 = [i for i in range(len(V)) if (V[i] > .9 and i % 5 == 0)]
            pos2 = [i for i in range(len(V)) if (V[i] <= .9)]
            position = concatenate((pos1, pos2))
            position = sort(position)
            time = time[position]
            I_exp = I_exp[position]
            V = V[position]
            
        kp0 =  0.005266845411489327
        kd0 =  102.4538696300204
        eta_p =  0.06665843701293933
        eta_d =  79.91922227381272
        g0 =  0.0020230656429231033
        g_min =  0.0013655460276781153
        g_max =  0.005124529803901193
        
    elif fname == 'example_file_2.txt':
        data = loadtxt(fdir+fname)
        title = 'EXAMPLE FILE 2'
        start = 0
        stop = -1
        time = data[start:stop:step, 0]
        V = data[start:stop:step, 2]
        I_exp = data[start:stop:step, 1]
            
        kp0 = 3.933e-8
        eta_p = 38
        kd0 = 7.5e-3
        eta_d = 0.03
        g0 = 8.988e-4
        g_min = 1.097e-3
        g_max = 14e-3 
    else:
        data = loadtxt(fdir+fname)
        title = 'NEW FILE'
        start = 0
        stop = -1
        time = data[start:stop:step, 0]
        V = data[start:stop:step, 2]
        I_exp = data[start:stop:step, 1]
        
        # Random parameters for undefined files
        kp0 = 1e-3
        eta_p = 1e-2
        kd0 = 10
        eta_d = 10
        g0 = 1e-3
        g_min = 1e-3
        g_max = 10e-3
        
    param = [kp0, eta_p, kd0, eta_d, g0, g_min, g_max]

    return time, V, I_exp, title, param
