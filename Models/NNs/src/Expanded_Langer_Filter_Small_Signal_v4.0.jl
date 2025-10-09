#Langer-Filter Signal
using Plots, Optim, BlackBoxOptim, LineSearches, FFTW, DSP, MATLAB, Polynomials, JLD
pyplot()

# save("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Langer_Signals_CST\\4_5_GHz\\ttt_sim.jld", "ttt_sim", ttt_sim)
# save("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Langer_Signals_CST\\4_5_GHz\\yyy_sim.jld","yyy_sim", yyy_sim)
# save("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Langer_Signals_CST\\4_5_GHz\\ttt_Gauss.jld", "ttt_Gauss", ttt_Gauss)
# save("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Langer_Signals_CST\\4_5_GHz\\yyy_Gauss.jld", "yyy_Gauss", yyy_Gauss)

ttt_sim = load("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Langer_Signals_CST\\4_5_GHz\\ttt_sim.jld")["ttt_sim"]
yyy_sim = load("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Langer_Signals_CST\\4_5_GHz\\yyy_sim.jld")["yyy_sim"]
ttt_Gauss = load("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Langer_Signals_CST\\4_5_GHz\\ttt_Gauss.jld")["ttt_Gauss"]
yyy_Gauss = load("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Langer_Signals_CST\\4_5_GHz\\yyy_Gauss.jld")["yyy_Gauss"]

# ttt_sim = ttt_sim[1:3:end]
# yyy_sim = yyy_sim[1:3:end]
# ttt_Gauss = ttt_Gauss[1:3:end]
# yyy_Gauss = yyy_Gauss[1:3:end]

plot(ttt_sim,yyy_sim, seriestype = :scatter)
plot!(ttt_Gauss,yyy_Gauss, seriestype = :scatter)

N = 500#250# 167#
stepstart = 1
const time_span = ttt_sim[stepstart:stepstart+N-1]
const time_span_Ext = ttt_sim[:]

#Original Data
data_original = yyy_sim[:]
data_training = yyy_sim[stepstart:stepstart+N-1]

#Function model -> our activation function of the hidden layer
Nb = 3

function model(param) # model for the original function
    result=0.0
    @inbounds for i in 1:Nb
        result = result.+harmsin(param[1+4*(i-1)],param[2+4*(i-1)],param[3+4*(i-1)],param[4*(i)])     # param[1+4*(i-1):4*(i)])
    end
    return result
end

function harmsin(A,ω,α,Φ)#param::Vector{Float64})
    return A.*sin.(ω.*time_span.+Φ).*exp.(-α.*time_span)
end

function modelEx(param)
    result=0.0
    @inbounds for i in 1:Nb
        result = result.+harmsinEx(param[1+4*(i-1)],param[2+4*(i-1)],param[3+4*(i-1)],param[4*(i)])
    end
    return result
end

function harmsinEx(A::Float64,ω::Float64,α::Float64,Φ::Float64)#param::Vector{Float64})
    return A.*sin.(ω.*time_span_Ext.+Φ).*exp.(-α.*time_span_Ext)
end

#Visualizing data
plot(ttt_sim, data_original,color=:blue, lw=1.5, label="Original", seriestype = :scatter)
plot!(time_span, data_training, lw=2.5, seriestype = :scatter, label="Training Points", color=:red)

#Impulse Response of the system - Data without the Gaussian-Modeled Sinus
plot!(ttt_Gauss,yyy_Gauss)

function loss_NN(param)
    loss = sum(abs2, conv(model(param), yyy_Gauss[1:N])[1:N].-data_training)
    return loss
end

function loss_NN2(param)
    if minimum(param[3:4:end])>0 && minimum(param[2:4:end])>0
    loss = sum(abs2, conv(model(param), yyy_Gauss[1:N])[1:N].-data_training)
    else
    loss = 100000
    end
    return loss
end

function NRMSE(param)
    loss = sqrt(sum(abs2, model(param).-data_training)./size(data_training,1))./(maximum(data_training)-minimum(data_training))
    return loss
end

function train_model()
    #GA1
    f_0 = 0.01
    f_end = 10^11
    A_0 = -10*maximum(data_training)
    A_max = 10*maximum(data_training)
    alpha_0 = 0
    alpha_max = 10^9
    Φ_0 = -2*π
    Φ_max = 2*π
    SearchRange =  [(A_0, A_max), (f_0, f_end),(alpha_0, alpha_max), (Φ_0,Φ_max)]
    for i in 1:Nb-1
        SearchRange = [SearchRange; [(A_0, A_max), (f_0, f_end),(alpha_0, alpha_max), (Φ_0,Φ_max)]]
    end
    res = bboptimize(loss_NN; SearchRange, NumDimensions = (Nb*4), Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = 10.0, PopulationSize = 50) #BlackBox Optimization (using BlackBoxOptim)! #SearchRange
    result_GA = best_candidate(res)
    #GA2
    f_0 = minimum(result_GA[2:4:end])/10
    f_end = maximum(result_GA[2:4:end])*10
    A_0 = minimum(result_GA[1:4:end])*10
    A_max = maximum(result_GA[1:4:end])*10
    alpha_0 = minimum(result_GA[3:4:end])/10
    alpha_max = maximum(result_GA[3:4:end])*10
    Φ_0 = minimum(result_GA[4:4:end])*10
    Φ_max = maximum(result_GA[4:4:end])*10
    SearchRange =  [(A_0, A_max), (f_0, f_end),(alpha_0, alpha_max), (Φ_0,Φ_max)]
    for i in 1:Nb-1
        SearchRange = [SearchRange; [(A_0, A_max), (f_0, f_end),(alpha_0, alpha_max), (Φ_0,Φ_max)]]
    end
    res2 = bboptimize(loss_NN; SearchRange, NumDimensions = (Nb*4), Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = 20.0, PopulationSize = 50) #BlackBox Optimization (using BlackBoxOptim)! #SearchRange
    result_GA2 = best_candidate(res2)
    #GA3
    if loss_NN(result_GA2) > 10^-10
        f_0 = minimum(result_GA[2:4:end])/10
        f_end = maximum(result_GA[2:4:end])*10
        A_0 = minimum(result_GA[1:4:end])*10
        A_max = maximum(result_GA[1:4:end])*10
        alpha_0 = minimum(result_GA[3:4:end])/10
        alpha_max = maximum(result_GA[3:4:end])*10
        Φ_0 = minimum(result_GA[4:4:end])*10
        Φ_max = maximum(result_GA2[4:4:end])*10
        SearchRange =  [(A_0, A_max), (f_0, f_end),(alpha_0, alpha_max), (Φ_0,Φ_max)]
        for i in 1:Nb-1
            SearchRange = [SearchRange; [(A_0, A_max), (f_0, f_end),(alpha_0, alpha_max), (Φ_0,Φ_max)]]
        end
        res3 = bboptimize(loss_NN,result_GA2; SearchRange, NumDimensions = (Nb*4), Method = :adaptive_de_rand_1_bin_radiuslimited,  MaxTime = 10.0, PopulationSize = 50) #BlackBox Optimization (using BlackBoxOptim)! #SearchRange
        result_GA2 = best_candidate(res3)
    end
    #NM
    if loss_NN(result_GA) < loss_NN(result_GA2)
        result_GA2 = result_GA[:]
    end
    result_final = optimize(loss_NN2,result_GA2, NelderMead(), Optim.Options(show_trace=true, iterations = 10000000, time_limit=10.0, g_tol=1e-32, allow_f_increases=true))
    best_param_fit = Optim.minimizer(result_final)
    if loss_NN(best_param_fit) > 10^-12
    result_final = optimize(loss_NN2,best_param_fit, NelderMead(), Optim.Options(show_trace=true, iterations = 10000000, time_limit=10.0, g_tol=1e-32, allow_f_increases=true))
    best_param_fit = Optim.minimizer(result_final)
    end
    return best_param_fit
end
#end

param_fit = train_model()
error("come back")
loss_NN(param_fit)
NRMSE(param_fit)

save("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Expanded_Method\\Best_Parameters\\param_fit.jld", "param_fit", param_fit)
param_fit = load("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Expanded_Method\\Best_Parameters\\param_fit.jld")["param_fit"]

#param_fit_sampling_1_2 = param_fit[:]
save("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Expanded_Method\\Best_Parameters\\param_fit_sampling_1_2.jld", "param_fit_sampling_1_2", param_fit_sampling_1_2)
param_fit_sampling_1_2 = load("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Expanded_Method\\Best_Parameters\\param_fit_sampling_1_2.jld")["param_fit_sampling_1_2"]

#param_fit_sampling_1_3 = param_fit[:]
save("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Expanded_Method\\Best_Parameters\\param_fit_sampling_1_3.jld", "param_fit_sampling_1_3", param_fit_sampling_1_3)
param_fit_sampling_1_3 = load("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Expanded_Method\\Best_Parameters\\param_fit_sampling_1_3.jld")["param_fit_sampling_1_3"]



function validation_plot(param)
    model_example = model(param)
    model_example_conv = conv(model_example, yyy_Gauss[1:N])
    #plot(ttt_Gauss[1:N],yyy_Gauss[1:N],label="Gauss Input")
    plot(time_span[1:N],model_example_conv[1:N], label="Conv. Signal")
    #plot(time_span[1:N],model_example[1:N], label="Non-Conv. Signal")
    scatter!(time_span[1:N],data_training[1:N], label="Training Data")
end
validation_plot(param_fit)

function validation_plotExt(param)
    model_example = modelEx(param)
    model_example_conv = conv(model_example, yyy_Gauss[:])
    #plot(ttt_Gauss[:],yyy_Gauss[:],label="Gauss Input")
    plot(time_span_Ext[:],model_example_conv[1:size(model_example,1)], label="Conv. Signal")
    plot!(time_span_Ext[:],model_example[:], label="Non-Conv. Signal")
    scatter!(time_span_Ext[:],yyy_sim[:], label="Full Data")
end
validation_plotExt(param_fit)

Results_time_Conv_RBFN = model_example_conv[1:size(model_example,1)]
Time_points_Conv_RBFN  =time_span_Ext[:]

Results_freq_Conv_RBFN = F_fit[1:end]
Freq_points_Conv_RBFN  = collect(freqzz[1:end])

save("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Expanded_Method\\Best_Parameters\\Results_time_Conv_RBFN.jld", "Results_time_Conv_RBFN", Results_time_Conv_RBFN)
Results_time_Conv_RBFN1 = load("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Expanded_Method\\Best_Parameters\\Results_time_Conv_RBFN.jld")["Results_time_Conv_RBFN"]
save("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Expanded_Method\\Best_Parameters\\Time_points_Conv_RBFN.jld", "Time_points_Conv_RBFN", Time_points_Conv_RBFN)
Time_points_Conv_RBFN1 = load("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Expanded_Method\\Best_Parameters\\Time_points_Conv_RBFN.jld")["Time_points_Conv_RBFN"]

save("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Expanded_Method\\Best_Parameters\\Results_freq_Conv_RBFN.jld", "Results_freq_Conv_RBFN", Results_freq_Conv_RBFN)
Results_freq_Conv_RBFN1 = load("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Expanded_Method\\Best_Parameters\\Results_freq_Conv_RBFN.jld")["Results_freq_Conv_RBFN"]
save("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Expanded_Method\\Best_Parameters\\Freq_points_Conv_RBFN.jld", "Freq_points_Conv_RBFN", Freq_points_Conv_RBFN)
Freq_points_Conv_RBFN1 = load("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Expanded_Method\\Best_Parameters\\Freq_points_Conv_RBFN.jld")["Freq_points_Conv_RBFN"]

#param_Conv_RBFN = best_param_fit[:]
save("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Expanded_Method\\Best_Parameters\\param_Conv_RBFN.jld", "param_Conv_RBFN", param_Conv_RBFN)
param_Conv_RBFN1 = load("C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Expanded_Method\\Best_Parameters\\param_Conv_RBFN.jld")["param_Conv_RBFN"]


##
#Fourier Transform
#Frequency Domain Analysis of the Signals - Fourier Transform
#Setting time parameters
tstart = ttt_sim[stepstart]
tend = ttt_sim[stepstart+N-1] # 6*N+N0/2= 6*40+75=315
Δt = (tend-tstart)/(N-1)

tstart = ttt_sim[stepstart+1000]
#Extrapolated signal in time
#conv(model(param), yyy_Gauss[1:N])[1:N]
param_fit = param_Conv_RBFN1[:]
sol_fit = conv(modelEx(param_fit), yyy_Gauss)[1:size(yyy_Gauss,1)]

#FFT of Original and Fitted Signals
F_fit = fft(sol_fit[stepstart:end]) |> fftshift
F_original = fft(data_original[stepstart:end]) |> fftshift
freqzz = fftfreq(length(time_span_Ext[stepstart:end]), 1/Δt) |> fftshift

F_fit = fft(sol_fit[stepstart+1000:end]) |> fftshift
F_original = fft(data_original[stepstart+1000:end]) |> fftshift
freqzz = fftfreq(length(time_span_Ext[stepstart+1000:end]), 1/Δt) |> fftshift


#Cut Signal (short) in Time
data_original_cut = data_original[stepstart+1000:stepstart+N-1]

#FFT of Cut Signal
F_cut = fft(data_original_cut) |> fftshift
freqzz_cut = fftfreq(length(ttt_sim[stepstart:stepstart+N-1]), 1/Δt) |> fftshift

#
freq_domain_fit = plot(freqzz[35500:38000], abs.(F_fit[35500:38000]), label = "Spec-Ext", color=:black,  lw=3, ylabel="Amp.", xlabel="Freq [Hz]",legendfontsize = 28, guidefont = (28), tickfont = (27))
plot!(freqzz[35500:38000], abs.(F_original)[35500:38000], label = "Spec-Orig (T_{fin})", color=:red,  lw=2, line = (:dashdot, 3),uselatex=true)
plot!(freqzz_cut, abs.(F_cut), label = "Spec-Orig (T_{max})", color=:cyan,  lw=1, line = (:dashdot, 3))

freq_domain_fit = plot(freqzz, abs.(F_fit), label = "Spec-Ext", color=:black,  lw=3, ylabel="Amp.", xlabel="Freq [Hz]",legendfontsize = 28, guidefont = (28), tickfont = (27))
plot!(freqzz, abs.(F_original), label = "Spec-Orig (T_{fin})", color=:red,  lw=2, line = (:dashdot, 3),uselatex=true)
plot!(freqzz_cut, abs.(F_cut), label = "Spec-Orig (T_{max})", color=:cyan,  lw=1, line = (:dashdot, 3))


freq_domain_fit = plot(freqzz, abs.(F_fit), label = "Spec-Ext", color=:black,  lw=3, ylabel="Amp.", xlabel="Freq [Hz]",legendfontsize = 28, guidefont = (28), tickfont = (27))
plot!(freqzz, abs.(F_original), label = "Spec-Orig (T_{fin})", color=:red,  lw=2, line = (:dashdot, 3),uselatex=true)



param_fit[3:4:end]'[2:3]

param_fit_conv = param_Conv_RBFN1[:]

Q1 = (param_fit_conv[2:4:end]')[:][6]/(2*param_fit_conv[3:4:end]'[6])

Q2 = (param_fit_conv[2:4:end]')[:][8]/(2*param_fit_conv[3:4:end]'[8])


Q1 = (param_fit_conv[2:4:end]')./(2*param_fit_conv[3:4:end]')

Q2 = (param_fit_conv[2:4:end]')./(2*param_fit_conv[3:4:end]')


f1 = param_fit_conv[6]/(2*π)
f2 = param_fit_conv[10]/(2*π)
Q1_ex = Q1[2]
Q2_ex = Q1[3]


Q1_ref = 320.1562
Q2_ref = 295.3006

f1_ref = 4.5592*10^9
f2_ref = 4.5346*10^9

Δf1 = abs(f1-f1_ref)/(f1_ref)
Δf2 = abs(f2-f2_ref)/(f2_ref)

ΔQ1 = abs(Q1_ex-Q1_ref)/(Q1_ref)
ΔQ2 = abs(Q2_ex-Q2_ref)/(Q2_ref)


#Paper Plotting
# plot(ttt_sim,yyy_sim, seriestype = :scatter)
# plot!(ttt_Gauss,yyy_Gauss, seriestype = :scatter)

# time_domain_fit=plot(ttt_sim*10^9, yyy_sim,lw=3.5, color=:green, ylabel="Amplitude [a.u.]", xlabel="t [ns]",legendfontsize = 23, guidefont = (23), tickfont = (22), xlims = (0,1.2e-7*10^9), xticks=(round.(0:2*1e-8*10^9:1e-7*10^9, digits=1)))

# time_domain_fit=plot(ttt_Gauss*10^9, yyy_Gauss,lw=3.5, color=:green, ylabel="Amplitude [a.u.]", xlabel="t [ns]",legendfontsize = 23, guidefont = (23), tickfont = (22), xlims = (0,1.20e-8*10^9), xticks=(round.(0:2*1.5e-9*10^9:1e-7*10^9, digits=1)))
