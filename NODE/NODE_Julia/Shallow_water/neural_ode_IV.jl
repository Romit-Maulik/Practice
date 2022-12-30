using DifferentialEquations, Flux, DiffEqFlux
using Plots, NPZ
using IterTools: ncycle
using BSON: @save, @load
include("./Custom_DataLoader.jl")
# using .Custom_DataLoader

# Julia starts from 1
true_data = npzread("SWE_RNN_Training_Data.npy") |>f32
true_data = permutedims(true_data, (1,3,2)) 
data_len = size(true_data)[3]
state_data = true_data[:,1:8,:] # last index inclusive

tspan = (0.0,true_data[1,9,data_len])
tsteps = 0.0:tspan[2]/data_len:tspan[2]
tsteps = tsteps[1:100]
T_ = Array{Float32, 3}(undef, size(true_data)[1], 1, size(true_data)[3])

# Outer For-loop
for i in 1:size(true_data)[1]          
    T_[i,1,:] = tsteps[:]
end

# print(size(T_),size(state_data))


test_data = npzread("SWE_RNN_Testing_Data.npy") |>f32
test_data = permutedims(test_data, (1,3,2))
u0 = test_data[1,1:8,1]

my_nn = Chain(
  Dense(8, 32, relu),
  Dense(32, 8)) |> f32

p, re = Flux.destructure(my_nn)

function right_hand_side(du,u,p,t)
  m = re(p)
  nn_output = m(u)
  du[1] = nn_output[1]
  du[2] = nn_output[2]
end

# prob = ODEProblem(right_hand_side,u0,tspan,p)
# sol = solve(prob, Tsit5())

# # Plot the solution
# plot(sol)
# savefig("Initial_NN_ODE.png")

function predict_ode(state_batch,time_batch)
  tmp_prob = remake(prob; p = p,u=state_batch[1])
  pred = Array(solve(tmp_prob, Tsit5(), saveat = time_batch))
  return pred
end

function loss_ode(state_batch,time_batch)
    pred = predict_ode(state_batch,time_batch)
    loss = sum(abs2, state_batch-pred)
    return loss
end

k = 20
train_loader = MyDataLoader((state_data,T_), batchsize = k, shuffle=true)
numEpochs = 200
losses=[]

# test_batch = iterate(train_loader)
# print(loss_ode(test_batch[1][1],test_batch[1][2]))
# print(typeof(loss_ode(test_batch[1][1],test_batch[1][2])))
# print(size(state_data))


cb() = begin
  
  l=loss_ode(test_data[1,1:8,:],tsteps)
  push!(losses, l)
  @show l

  # pred = predict_ode(state_data,tsteps)
  
  # plt = plot(x_data,hcat(transpose(pred),state_data), 
  #   line=(4, [:solid :solid :dash :dash]), ylim = (-6, 6),
  #   label=["y1-pred" "y2-pred" "y1-true" "y2-true"])
  
  # display(plt)
  # Tell sciml_train to not halt the optimization. If return true, then
  # optimization stops.
  return false
end

opt=ADAM(0.01)
Flux.train!(loss_ode, Flux.params(p), ncycle(train_loader,numEpochs), opt)#, cb=cb) # cb=Flux.throttle(cb, 10)

# # my_nn_final = re(p)
# # @save "mymodel.bson" my_nn_final

# # Inference
# @load "mymodel.bson" my_nn_final
# p, re = Flux.destructure(my_nn_final)


# prob_final = ODEProblem(right_hand_side, u0, tspan, p)
# sol_final = solve(prob_final,Tsit5())
# plot(sol_final)
# savefig("NN_ode_optimized.png") # Both state components close to 1