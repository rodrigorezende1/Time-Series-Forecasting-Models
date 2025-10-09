#Langer-Filter Signal
using Plots, Optim, BlackBoxOptim, LineSearches, FFTW, DSP, MATLAB
using LatexStrings
pyplot()
@mput
mat"
cd 'C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Prony_MatLab'
[data,msg] = fopen('signal4-6.txt','rt');
assert(data>=3,msg) % ensure the file opened correctly.
output = textscan(data,'%f%f%f');
fclose(data);
ttt= output{1};
yyy = output{2};
"
@mget ttt yyy

plot(ttt,yyy, seriestype = :scatter)

N = 1000
stepstart = 200
N_init = N
stepstart_init = stepstart
tstart = ttt[stepstart]
tend = ttt[stepstart+N-1] # 6*N+N0/2= 6*40+75=315
Δt = (tend-tstart)/(N-1) #8 GHz is the maximum Frequency)
const time_span = ttt[stepstart:stepstart+N-1]


#Original Data
data_original = yyy[:]
data_training = yyy[stepstart:stepstart+N-1]

#Function model -> our activation function of the first layer
Nb=10

function model(param)
    result=0.0
    @inbounds for i in 1:Nb
        result = result.+harmsin(param[1+4*(i-1)],param[2+4*(i-1)],param[3+4*(i-1)],param[4*(i)])     # param[1+4*(i-1):4*(i)])
    end
    return result
end

function harmsin(A::Float64,ω::Float64,α::Float64,Φ::Float64)#param::Vector{Float64})
    return A.*sin.(ω.*time_span.+Φ).*exp.(-α.*time_span)
end

function harmsin(A,ω,α,Φ)#param::Vector{Float64})
    return A.*sin.(ω.*time_span.+Φ).*exp.(-α.*time_span)
end
#Extrapolating

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
plot(ttt, data_original,color=:blue, lw=1.5, label="Original", seriestype = :scatter)

plot!(time_span, data_training, lw=2.5, seriestype = :scatter, label="Training Points", color=:red)

function loss_NN(param)
    loss = sum(abs2, model(param).-data_training)
    return loss
end

function loss_NN2(param)
    if minimum(param[3:4:end])>0 && minimum(param[2:4:end])>0
    loss = sum(abs2, model(param).-data_training)
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
    res = bboptimize(loss_NN; SearchRange, NumDimensions = (Nb*4), Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = 100.0, PopulationSize = 50) #BlackBox Optimization (using BlackBoxOptim)! #SearchRange
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
    res2 = bboptimize(loss_NN; SearchRange, NumDimensions = (Nb*4), Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = 200.0, PopulationSize = 50) #BlackBox Optimization (using BlackBoxOptim)! #SearchRange
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
        res3 = bboptimize(loss_NN,result_GA2; SearchRange, NumDimensions = (Nb*4), Method = :adaptive_de_rand_1_bin_radiuslimited,  MaxTime = 400.0, PopulationSize = 50) #BlackBox Optimization (using BlackBoxOptim)! #SearchRange
        result_GA2 = best_candidate(res3)
    end
    #NM
    if loss_NN(result_GA) < loss_NN(result_GA2)
        result_GA2 = result_GA[:]
    end
    result_final = optimize(loss_NN2,result_GA2, NelderMead(), Optim.Options(show_trace=true, iterations = 10000000, time_limit=200.0, g_tol=1e-32, allow_f_increases=true))
    best_param_fit = Optim.minimizer(result_final)
    if loss_NN(best_param_fit) > 10^-12
    result_final = optimize(loss_NN2,best_param_fit, NelderMead(), Optim.Options(show_trace=true, iterations = 10000000, time_limit=200.0, g_tol=1e-32, allow_f_increases=true))
    best_param_fit = Optim.minimizer(result_final)
    end
    return best_param_fit
end
#end

param_fit = train_model()
error("come back")
loss_NN(param_fit)
NRMSE(param_fit)

# param_fit_best = param_fit[:]
#Best value until now

#save("C:\\Programs\\PL\\Julia_Files\\Harmonic_Inversion\\param_fit_best.jld", "param_fit_best", param_fit_best)
# param_fit_best = load("C:\\Programs\\PL\\Julia_Files\\Harmonic_Inversion\\param_fit_best.jld")["param_fit_best"]
#param_fit_best_10 = load("C:\\Programs\\PL\\Julia_Files\\Harmonic_Inversion\\param_fit_best_10.jld")["param_fit_best_10"]
# loss_NN(param_fit_best_10)
# NRMSE(param_fit_best_10)


time_span_Ext=ttt[stepstart:end]#(4/3)*tend
#time_span=ttt[stepstart:stepstart+N-1]
function validation_plot(param)
    lay = @layout [ a; b]
    sol_fit = model(param)
    #sol_fit_Ext = modelEx(param)
    #Δf_f_t= sum(abs.(sol_fit_Ext-data_original[stepstart:end]))
    Δf_f_t= sum(abs.(sol_fit-data_original[stepstart:stepstart+N-1]))
    println("The Σ(abs(y-y_fit))=$(Δf_f_t)")
    p1=plot(time_span, sol_fit,lw=3.5, color=:green,label="Fitted")
    #plot!(p1,ttt, data_original,lw=1.0, color=:blue, seriestype = :scatter, label="Original")
    plot!(p1,time_span, data_training,lw=1.0, color=:blue, seriestype = :scatter, label="Original")
    #p2=plot(time_span_Ext, sol_fit-data_original[stepstart:end], label="Difference")
    p2=plot(time_span, sol_fit-data_original[stepstart:stepstart+N-1], label="Difference")
    plot(p1,p2, layout=lay)
end
validation_plot(param_fit)

function validation_plotExt(param)
    lay = @layout [ a; b]
    sol_fit_Ext = modelEx(param)
    Δf_f_t= sum(abs.(sol_fit_Ext-data_original[stepstart:end]))
    println("The Σ(abs(y-y_fit))=$(Δf_f_t)")
    p1=plot(time_span_Ext, sol_fit_Ext,lw=3.5, color=:green,label="Fitted")
    plot!(p1,ttt, data_original,lw=1.0, color=:blue, seriestype = :scatter, label="Original")
    p2=plot(time_span_Ext, sol_fit_Ext-data_original[stepstart:end], label="Difference")
    plot(p1,p2, layout=lay)
end
validation_plotExt(param_fit)



##
#Fourier Transform
#Pronys time
N = 500
stepstart = 1000
tstart = ttt[stepstart]
tend = ttt[stepstart+N-1] # 6*N+N0/2= 6*40+75=315
Δt = (tend-tstart)/(N-1)
#Signal in time
time_span_Ext=ttt[stepstart:end]

sol_fit = modelEx(param_fit)
Δf_f_t = sum(abs.(sol_fit-data_original[stepstart:end]))

F_fit = fft(sol_fit) |> fftshift
F_original = fft(data_original[stepstart:end]) |> fftshift
freqzz = fftfreq(length(time_span_Ext), 1/Δt) |> fftshift

#Cut Singal in Time
data_original_cut = data_original[stepstart:stepstart+N-1]

F_cut = fft(data_original_cut) |> fftshift
freqzz_cut = fftfreq(length(ttt[stepstart:stepstart+N-1]), 1/Δt) |> fftshift


##
#Plotting the extrapolated original signal and spectrum and pronys

time_span_10Ext = collect(time_span_Ext[1]:Δt:7*time_span_Ext[end])
# time_span_10Ext = ttt_long[:]

function modelNbEx(param)
    result=0.0
    @inbounds for i in 1:Nb
        result = result.+harmsinNbEx(param[1+4*(i-1)],param[2+4*(i-1)],param[3+4*(i-1)],param[4*(i)])     # param[1+4*(i-1):4*(i)])
    end
    return result
end

function harmsinNbEx(A::Float64,ω::Float64,α::Float64,Φ::Float64)#param::Vector{Float64})
    return A.*sin.(ω.*time_span_10Ext.+Φ).*exp.(-α.*time_span_10Ext)
end

sol_fit_NbExt = modelNbEx(param_fit)
p1=plot(time_span_10Ext, sol_fit_NbExt,lw=3.5, color=:green,label="Fitted")

F_Ext = fft(sol_fit_NbExt) |> fftshift
freqzz_Ext = fftfreq(length(sol_fit_NbExt), 1/Δt) |> fftshift
#freq
plot(freqzz_Ext,abs.(F_Ext),label = "Spec-Fitted")


@mput
mat"
cd 'C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Prony_MatLab'
[data,msg] = fopen('signal4-6_longSignal.txt','rt');
assert(data>=3,msg) % ensure the file opened correctly.
output = textscan(data,'%f%f%f');
fclose(data);
ttt_long = output{1};
yyy_long = output{2};
"
@mget ttt_long yyy_long

Δt_long = (ttt_long[end]-ttt_long[1])/length(yyy_long)

F_Original_Ext = fft(yyy_long[1000:end]) |> fftshift
freqzz_Original_Ext = fftfreq(length(yyy_long[1000:end]), 1/Δt_long) |> fftshift
#freq
#Complete Fitting
time_span_plot = collect(ttt_long[stepstart_init]:Δt:ttt_long[end])
function modelNbExPlot(param)
    result=0.0
    @inbounds for i in 1:Nb
        result = result.+harmsinNbExPlot(param[1+4*(i-1)],param[2+4*(i-1)],param[3+4*(i-1)],param[4*(i)])     # param[1+4*(i-1):4*(i)])
    end
    return result
end

function harmsinNbExPlot(A::Float64,ω::Float64,α::Float64,Φ::Float64)#param::Vector{Float64})
    return A.*sin.(ω.*time_span_plot.+Φ).*exp.(-α.*time_span_plot)
end
sol_fit_NbExtPlot = modelNbExPlot(param_fit)


@mput
mat"
cd 'C:\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\Prony_MatLab'
load('prony_output.mat')
fp = linspace(FIL.f1,FIL.f2,2126);
Yp = prony_spectrum(FIL,fp);
t_prony = FIL.t_prony
y_prony = FIL.y_prony
"
@mget fp Yp t_prony y_prony

freq_prony = fp[:]
Yfreq_pronyabs=abs.(Yp[:])

# time_domain_fit=plot(ttt_long[1000:end], yyy_long[1000:end],  lw=3.5, label = "Orig", color=:green,markersize = 5, 	xlims = (1.2e-8,1e-7), xticks=(round.(1.34e-8:2e-8:1e-7, digits=10)))
# plot!(time_span_10Ext[1000:end], sol_fit_NbExt[1000:end],markersize = 20,lw=10, color=:black,seriestype = :scatter,m = [:+],label="Ext", ylabel="Amp.", xlabel="s",legendfontsize = 28, guidefont = (28), tickfont = (27))
# plot!(time_span_10Ext[1:10:1000], sol_fit_NbExt[1:10:1000],markersize = 12,lw=2.5, color=:black,seriestype = :scatter,m = [:.],label="Fit")
# plot!(t_prony[:], y_prony[:],  lw=2.5, label = "k=1000K=2493K=3985K=5477K= 6910", color=:red, m = [:+],seriestype = :scatter,markersize = 17)#, xaxis = ((tstart,6*tend), tstart:2.5*10^-7:6*tend))
# y1=[-0.08, 0.08]
# x1 = [ttt[stepstart_init+N_init-1], ttt[stepstart_init+N_init-1]]
# plot!(x1,y1, line = (:dashdot, 3), color=:cyan,lw=10)#, label = "Tr. Data Lim.")
time_domain_fit=plot(time_span_plot, sol_fit_NbExtPlot,lw=3.5, color=:green,label="Ext", ylabel="Amplitude", xlabel="s",legendfontsize = 23, guidefont = (23), tickfont = (22), xlims = (1.0e-14,2.8e-7), xticks=(round.(1.0e-8:5e-8:2.8e-7, digits=9)))
plot!(ttt_long[1000:4:end], yyy_long[1000:4:end],  lw=2, label = "Orig", color=:blue, seriestype = :scatter,markersize = 5)
plot!(t_prony[:], y_prony[:],  lw=1.5, label = "Prony", color=:red, m = [:+],seriestype = :scatter,markersize = 10)#, xaxis = ((tstart,6*tend), tstart:2.5*10^-7:6*tend))
y1=[-0.08, 0.08]
x1 = [ttt[stepstart_init+N_init-1], ttt[stepstart_init+N_init-1]]
x2 = [ttt[stepstart_init], ttt[stepstart_init]]
y1=[-0.08, 0.08]
y2=[-0.2, 0.2]
plot!(x1,y1, line = (:dashdot, 3), color=:cyan)#, label = "Tr. Data Lim.")
plot!(x2,y2, line = (:dashdot, 3), color=:cyan)#, label = "Tr. Data Lim.")


# time_domain_fit=plot(time_span_plot, sol_fit_NbExtPlot,lw=3.5, color=:green,label="Ext", ylabel="Amplitude", xlabel="s",legendfontsize = 23, guidefont = (23), tickfont = (22), xlims = (1.0e-14,2.8e-7), xticks=(round.(1.0e-8:5e-8:2.8e-7, digits=9)))
# plot!(ttt_long[1000:4:end], yyy_long[1000:4:end],  lw=2, label = "Orig", color=:blue, seriestype = :scatter,markersize = 5)
# plot!(t_prony[:], y_prony[:],  lw=1.5, label = "Prony", color=:red, m = [:+],seriestype = :scatter,markersize = 10)#, xaxis = ((tstart,6*tend), tstart:2.5*10^-7:6*tend))
# y1=[-0.08, 0.08]
# x1 = [ttt[stepstart_init+N_init-1], ttt[stepstart_init+N_init-1]]
# x2 = [ttt[stepstart_init], ttt[stepstart_init]]
# y1=[-0.08, 0.08]
# y2=[-0.2, 0.2]
# plot!(x1,y1, line = (:dashdot, 3), color=:cyan)#, label = "Tr. Data Lim.")
# plot!(x2,y2, line = (:dashdot, 3), color=:cyan)#, label = "Tr. Data Lim.")


freq_domain_fit = plot(freqzz_Ext[11563:11844], abs.(F_Ext)[11563:11844], label = "Spec-Ext", color=:black,  lw=7.5, ylabel="Amp.", xlabel="Freq [Hz]",legendfontsize = 28, guidefont = (28), tickfont = (27), ylims = (0,40), xlims = (4.35e9,4.75e9), xticks=(round.(4.35e9:1.25e8:4.75e9, digits=2)))
plot!(freqzz_Original_Ext[9165:9387], abs.(F_Original_Ext)[9165:9387], label = L"Spec-Orig ($T_{fin}$)", color=:green,  lw=5, line = (:dashdot, 8),uselatex=true)
plot!(freq_prony[1:1065],Yfreq_pronyabs[1:1065], color=:red, label = "Spec-Prony",  lw=7.5, line = (:dot, 8))
plot!(freqzz[1178:1207], abs.(F_original[1178:1207]), label = L"Spec-Orig ($T_{max}$)", color=:cyan,  lw=5, line = (:dashdot, 7))

l1 = @layout [a ; b]

plot(time_domain_fit, freq_domain_fit, layout = l1)


#Extrapolation error
function NRMSEI(Yₜₑₛ::Vector{Float64},yₑ::Vector{Float64})
    Nₜₛ = size(Yₜₑₛ,1) # Nₜₛ - Number of test scenarios
    Nₒᵤₜ = size(Yₜₑₛ,2) #Nₒᵤₜ - Number of outputs pro scenario
    Mat² = (Yₜₑₛ-yₑ).^2
    Vect² = sum(Mat²)./Nₜₛ
    D = sum(Vect²)/Nₒᵤₜ
    rmse = (D)^(1/2)
    range_Yₜₑₛ = maximum(Yₜₑₛ)-minimum(Yₜₑₛ)
    #if size(Yₜₑₛ,1) > 2 || size(Yₜₑₛ,2) > 2
    nmrse = rmse/range_Yₜₑₛ
    #else
    #nmrse = rmse
    #end
    return nmrse
end
Error_Ext = NRMSEI(yyy_long[2000:end], sol_fit_NbExt[2000:end])

#Calculation of ω and Q=ω/(2α)
(param_fit[2:4:end]'./(2*π))[:][6]
(param_fit[2:4:end]'./(2*π))[:][8]

param_fit[3:4:end]'[2:3]

Q1 = (param_fit[2:4:end]')[:][6]/(2*param_fit[3:4:end]'[6])

Q2 = (param_fit[2:4:end]')[:][8]/(2*param_fit[3:4:end]'[8])
