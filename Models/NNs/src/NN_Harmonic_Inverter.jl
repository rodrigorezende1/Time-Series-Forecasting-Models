using Plots, Optim, BlackBoxOptim, LineSearches

#Original signal constants
A1_or, ω1_or, α1_or = [3.75, 2.93*10^4, 2.74]
A2_or, ω2_or, α2_or = [2.13, 4.25*10^4, 124.27]
A3_or, ω3_or, α3_or = [0.87, 3.84*10^4, 46.87]

original_param = [A1_or, ω1_or, α1_or, A2_or, ω2_or, α2_or, A3_or, ω3_or, α3_or]

#Training data time span
tstart = 0.0
tend = (3/4)*1.0 #the original simulaiton will be tstart:Δt:(4\3)*tend
Δt = 0.0005
time_span=tstart:Δt:tend
#time_span = 10 .^(range(-3,stop=-0.15,length=1001))

#Function model -> our activation function of the first layer
function model(param)
    A1, ω1, α1, A2, ω2, α2, A3, ω3, α3 = param
    return harmsin(A1, ω1, α1).+harmsin(A2, ω2, α2).+harmsin(A3, ω3, α3)
end

function harmsin(A,ω,α)
    return A.*sin.(ω.*time_span).*exp.(-α.*time_span)
end

#Original complete data
time_span = tstart:Δt:(4/3)*tend
data_original = model(original_param)
plot(time_span, data_original,color=:blue, lw=1.5)

#Cut/ Training data
time_span = tstart:Δt:tend
data = model(original_param)
plot(time_span, data, color=:green, lw=2.5, seriestype = :scatter)

#Noisy data
# σN = 0.1
# data_noise = data_noise + σN*randn(size(data))

#Definig the loss function
function loss_NN(param)
        # f_t = model(param)
        #lambda=0.0 # regularization term
        #loss = log(sum(abs.(model(param).-data)))
        loss = sum(abs2, model(param).-data)#+lambda*sum(param)
        #loss = sum((model(param).-data).^2)
        return loss
end


#Train the model
#initial_x = [1, 1*10^4, 10^-1, 2.5, 2.5*10^4, 2.5*10^-1, 3.3, 3.3*10^4, 3.3*10^-1]
initial_x = best_param_fit#[5, 1*10^5, 1,5, 1*10^5, 1, 5, 1*10^5, 1].*rand(9)
lower = [0,0,0,0,0,0,0,0,0]
upper = [10,10^12,10,10,10^12,10,10,10^12,10]
function train_model()
    println("The initial loss function is $(loss_NN(initial_x)[1])")
    inner_optimizer = LBFGS(; m = 100,
         alphaguess = LineSearches.InitialStatic(),
        linesearch=LineSearches.BackTracking())
    result = optimize(loss_NN, lower, upper, initial_x, Fminbox(inner_optimizer), Optim.Options(show_trace=true, iterations = 100000, time_limit=200.0, g_tol=1e-32, allow_f_increases=true))
    return result
end

param_fit = train_model()
#best_param_fit = abs.(Optim.minimizer(param_fit))
best_param_fit = abs.(best_candidate(param_fit))

time_span=tstart:Δt:(4/3)*tend
function validation_plot(param)
    sol_fit = model(param)
    plot(time_span, sol_fit,lw=3.5, color=:green)
    plot!(time_span, data_original,lw=1.0, color=:blue, seriestype = :scatter)
    Δf_f_t= sum(abs.(sol_fit-data_original))
    println(Δf_f_t)
end

validation_plot(best_param_fit)

Δdiff=abs.(original_param-best_param_fit)./original_param
