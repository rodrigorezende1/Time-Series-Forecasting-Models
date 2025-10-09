using Plots, Optim, BlackBoxOptim, LineSearches, FFTW, DSP, JLD

pyplot()
#Original signal constants
Nb = 5 #Number of basis function of the original function
Nb2 = 7 #Number of basis function desconsidering the Gaussian-Modulated Sine

Amax=1
Fmax=10^10
αmax=1*10^8
Φmax=2*π
parammax = [Amax; Fmax; αmax; Φmax]
original_param = rand(4,1)[:].*parammax
for i in 1:Nb-1
    original_param = [original_param; rand(4,1).*parammax]
end

#save("C:\\Programs\\PL\\Julia_Files\\Harmonic_Inversion\\original_param.jld", "original_param", original_param)
# original_param = load("\\Users\\TET1\\Desktop\\my_files\\Harmonic_Inversion\\First_Method\\original_param.jld")["original_param"]


#Training data time span
tstart = 0.0
tend = 4.0*10^-8# (3/4)*1.0 #the original simulaiton will be tstart:Δt:(4\3)*tend
Δt =  1/(3*maximum(original_param)) #at least 1/(3*4.25*10^10)
const time_span = collect(tstart:Δt:tend)
stepstart = 400
time_span2 = time_span[stepstart:end]
#time_span = 10 .^(range(-3,stop=-0.15,length=1001))

function model(param) # model for the original function
    result=0.0
    @inbounds for i in 1:Nb
        result = result.+harmsin(param[1+4*(i-1)],param[2+4*(i-1)],param[3+4*(i-1)],param[4*(i)])     # param[1+4*(i-1):4*(i)])
    end
    return result
end

function harmsin(A,ω,α,Φ)#param::Vector{Float64})
    return A.*sin.(ω.*time_span.+Φ).*exp.(-α.*time_span)
    #return A.*sin.(abs(ω).*time_span.+abs(Φ)).*exp.(-abs(α).*time_span)
end

function model2(param) # model for the function desconsidering the Gaussian-Modulated Sine
    result=0.0
    @inbounds for i in 1:Nb2
        result = result.+harmsin2(param[1+4*(i-1)],param[2+4*(i-1)],param[3+4*(i-1)],param[4*(i)])     # param[1+4*(i-1):4*(i)])
    end
    return result
end

function harmsin2(A,ω,α,Φ)#param::Vector{Float64})
    return A.*sin.(ω.*time_span2.+Φ).*exp.(-α.*time_span2)
    #return A.*sin.(abs(ω).*time_span.+abs(Φ)).*exp.(-abs(α).*time_span)
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

function modelEx2(param)
    result=0.0
    @inbounds for i in 1:Nb2
        result = result.+harmsinEx2(param[1+4*(i-1)],param[2+4*(i-1)],param[3+4*(i-1)],param[4*(i)])
    end
    return result
end

function harmsinEx2(A::Float64,ω::Float64,α::Float64,Φ::Float64)#param::Vector{Float64})
    return A.*sin.(ω.*time_span_Ext.+Φ).*exp.(-α.*time_span_Ext)
end

#Impulse Response of the system - Data without the Gaussian-Modeled Sinus
time_span_Ext = tstart:Δt:5.0*10^-7#(4/3)*tend
data_original = modelEx(original_param)
data_training = model(original_param)


plot(time_span_Ext, data_original,color=:blue, lw=1.5, label="Original",ticks = :native)
plot!(time_span, data_training ,color=:green, lw=1.5, label="Training",ticks = :native,seriestype = :scatter)
#Adding the impulse signal Gaussian-Modeled Sinus
function Gaussian_Sine_Ext(C,D,E,F,G)
    return C*sin.(D.*(E.-F.*time_span_Ext)).*exp.((-(E.-F.*time_span_Ext).^2)./G)
end

function Gaussian_Sine(C,D,E,F,G)
    return C*sin.(D.*(E.-F.*time_span)).*exp.((-(E.-F.*time_span).^2)./G)
end

C = 1
D = 2.1
E = 13.844651193682289
F = 3.2e9 #2.2 gleiche Δt
G = 20

GS_Ext = Gaussian_Sine_Ext(C,D,E,F,G)
GS = Gaussian_Sine(C,D,E,F,G)
#convolution between original and GS

cv_Ext = conv(GS_Ext,data_original)
cv_training = conv(GS,data_training)

plot(time_span_Ext, cv_Ext[1:size(time_span_Ext,1)],color=:green, lw=1.5, label="Convolution Ext")
plot!(time_span_Ext, data_original,color=:blue, lw=1.5, label="Impulse Response",ticks = :native)
plot!(time_span_Ext, GS_Ext,color=:red, lw=1.5, label="Gauss-Sine")



#Cut/ Training data of the original signal
conv_training = cv_training[1:size(time_span,1)]
plot!(time_span[1:end], conv_training[1:end],color=:green,seriestype = :scatter, lw=1.5, label="Convolution Training")

#Cut/ Training data  for the function desconsidering the Gaussian-Modulated Sine
noconv_training = cv_training[stepstart:size(time_span,1)]
plot!(time_span[stepstart:end], noconv_training, color=:cyan, markersize = 10, m = [:+],seriestype = :scatter, label="Training Points Not Conv")

plot!([time_span[end], time_span[end]], [-10, 10], color=:black, lw=3, label="Training Window")


#Definig the loss function
function loss_NN(param)
        guess = conv(GS,model(param))[1:size(time_span,1)]
        loss = sum(abs2, guess[1:size(time_span,1)].-conv_training[1:end])
        return loss
end

function loss_NN2(param)
        loss = sum(abs2, model2(param).-noconv_training)
        return loss
end


function loss_NNNM(param)
    if minimum(param[3:4:end])>0 && minimum(param[2:4:end])>0
        guess = conv(GS,model(param))[1:size(time_span,1)]
        loss = sum(abs2, guess[1:size(time_span,1)].-conv_training[1:end])
        return loss
    else
    loss = 100000000
    end
    return loss
end

function loss_NNNM2(param)
    if minimum(param[3:4:end])>0 #&& minimum(param[2:4:end])>0
    loss = sum(abs2, model2(param).-noconv_training)
    else
    loss = 100000000
    end
    return loss
end


function NRMSE(param)
        loss = sqrt(sum(abs2, model(param).-data_training)./size(data_training,1))./(maximum(data_training)-minimum(data_training))
        return loss
end
#Train the model

function train_model()
    #GA1
    f_0 = 0.01
    f_end = 10^11
    A_0 = -maximum(conv_training)
    A_max = maximum(conv_training)
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
    res2 = bboptimize(loss_NN,result_GA; SearchRange, NumDimensions = (Nb*4), Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = 100.0, PopulationSize = 50) #BlackBox Optimization (using BlackBoxOptim)! #SearchRange
    result_GA2 = best_candidate(res2)
    #NM
    if loss_NN(result_GA) < loss_NN(result_GA2)
        result_GA2 = result_GA[:]
    end
    result_final = optimize(loss_NNNM,result_GA2, NelderMead(), Optim.Options(show_trace=true, iterations = 10000000, time_limit = 50.0, g_tol=1e-32, allow_f_increases=true))
    best_param_fit = Optim.minimizer(result_final)
    loss_NN(best_param_fit)
    NRMSE(best_param_fit)
    return best_param_fit
end

param_fit = train_model()

loss_NN(param_fit)

function train_model2()
    #GA1
    f_0 = 0.01
    f_end = 10^11
    A_0 = -maximum(conv_training)
    A_max = maximum(conv_training)
    alpha_0 = 0
    alpha_max = 10^9
    Φ_0 = -2*π
    Φ_max = 2*π
    SearchRange =  [(A_0, A_max), (f_0, f_end),(alpha_0, alpha_max), (Φ_0,Φ_max)]
    for i in 1:Nb2-1
        SearchRange = [SearchRange; [(A_0, A_max), (f_0, f_end),(alpha_0, alpha_max), (Φ_0,Φ_max)]]
    end
    res = bboptimize(loss_NN2; SearchRange, NumDimensions = (Nb2*4), Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = 50.0, PopulationSize = 50) #BlackBox Optimization (using BlackBoxOptim)! #SearchRange
    result_GA = best_candidate(res)
    #GA2
    f_0 = minimum(result_GA[2:4:end])/10
    f_end = maximum(result_GA[2:4:end])*10
    A_0 = -abs(minimum(result_GA[1:4:end]))*10
    A_max = maximum(abs.(result_GA[1:4:end]))*10
    alpha_0 = minimum(result_GA[3:4:end])/10
    alpha_max = maximum(result_GA[3:4:end])*10
    Φ_0 = minimum(result_GA[4:4:end])*10
    Φ_max = maximum(abs.(result_GA[4:4:end]))*10
    SearchRange =  [(A_0, A_max), (f_0, f_end),(alpha_0, alpha_max), (Φ_0,Φ_max)]
    for i in 1:Nb-1
        SearchRange = [SearchRange; [(A_0, A_max), (f_0, f_end),(alpha_0, alpha_max), (Φ_0,Φ_max)]]
    end
    res2 = bboptimize(loss_NN2,result_GA; SearchRange, NumDimensions = (Nb2*4), Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = 50.0, PopulationSize = 50) #BlackBox Optimization (using BlackBoxOptim)! #SearchRange
    result_GA2 = best_candidate(res2)
    #NM
    if loss_NN2(result_GA) < loss_NN2(result_GA2)
        result_GA2 = result_GA[:]
    end
    result_final = optimize(loss_NNNM2,result_GA2, NelderMead(), Optim.Options(show_trace=true, iterations = 10000000, time_limit = 50.0, g_tol=1e-32, allow_f_increases=true))
    best_param_fit = Optim.minimizer(result_final)
    loss_NN(best_param_fit)
    NRMSE(best_param_fit)
    return best_param_fit
end

#end

param_fit2 = train_model2()

loss_NN2(param_fit2)


#Plotting the conv model
model_calc =  modelEx(param_fit)[stepstart:end]
cv_model = conv(GS_Ext,model_calc)
plot(time_span_Ext, cv_Ext[1:size(time_span_Ext,1)],color=:green, lw=1.5, label="Convolution",seriestype = :scatter)
plot!(time_span_Ext, cv_model[1:size(time_span_Ext,1)],color=:red, lw=1.5, label="Convolution Guess")


model_calc2 =  modelEx2(result_GA)
# cv_model = conv(GS_Ext,model_calc)
plot(time_span_Ext, cv_Ext[1:size(time_span_Ext,1)],color=:green, lw=1.5, label="Convolution",seriestype = :scatter)
plot!(time_span[stepstart:end], noconv_training, color=:cyan, markersize = 10, m = [:+],seriestype = :scatter, label="Training Points Not Conv")
plot!(time_span_Ext, model_calc2,color=:red, lw=1.5, label="Convolution Guess")



#Plotting the non conv model


#Δxdiff=abs.(original_param-param_fit)./original_param
#Δxdiff = abs.([original_param[7:9]; original_param[1:3]; original_param[4:6]]-param_fit)./original_param

time_span=tstart:Δt:40*tend#(4/3)*tend
data_original = model(original_param)
Δxdiff=abs.(original_param-param_fit)./original_param
# time_span = tstart:Δt:(6/3)*tend
# data_original = model(original_param)
function validation_plot(param)
    lay = @layout [ a; b]
    sol_fit = model(param)
    Δf_f_t= sum(abs.(sol_fit-data_original))
    println("The Σ(abs(y-y_fit))=$(Δf_f_t)")
    p1=plot(time_span, sol_fit,lw=3.5, color=:green,label="Fitted")
    plot!(p1,time_span, data_original,lw=1.0, color=:blue, seriestype = :scatter, label="Original")
    p2=plot(time_span, sol_fit-data_original, label="Difference")
    plot(p1,p2, layout=lay)
end

validation_plot(param_fit)

##
#Fourier Transform

#Signal in time
#time_span=tstart:Δt:3*tend#(4/3)*tend
time_span = tstart:Δt:6*tend

sol_fit = model(param_fit)
data_original = model(original_param)
Δf_f_t= sum(abs.(sol_fit-data_original))

F_fit = fft(sol_fit) |> fftshift
F_original = fft(data_original) |> fftshift

freqz = fftfreq(length(time_span), 1/Δt) |> fftshift

time_domain_fit = plot(time_span, sol_fit, label = "Fitted", color=:green, xaxis = ((tstart,6*tend), tstart:2.5*10^-7:6*tend))
time_domain_original = plot(time_span, data_original, label = "Original", xaxis = ((tstart,6*tend), tstart:2.5*10^-7:6*tend))
freq_domain_fit = plot(freqz, abs.(F_fit), label = "Spec-Fitted", color=:green)
freq_domain_original = plot(freqz, abs.(F_original), label = "Spec-Orig")
plot(time_domain_fit, freq_domain_fit, time_domain_original, freq_domain_original, layout = 4)

#Cut Singal in Time
time_span=tstart:Δt:tend
data_original_cut = model(original_param)

F_cut = fft(data_original_cut) |> fftshift
freqz_cut = fftfreq(length(time_span), 1/Δt) |> fftshift

time_domain_cut = plot(time_span, data_original_cut, xaxis = ((tstart,6*tend), tstart:2.5*10^-7:6*tend), label = "Cut (Training)", color=:red)
freq_domain_cut = plot(freqz_cut, abs.(F_cut), label = "Spec-Cut", color=:red)
plot(time_domain_cut, freq_domain_cut, layout = 2)

l = @layout [a  b; c  d; e  f]
plot(time_domain_fit, freq_domain_fit, time_domain_original, freq_domain_original, time_domain_cut, freq_domain_cut, layout = l)
