using DifferentialEquations, Flux, DiffEqFlux
using Plots, NPZ
using IterTools: ncycle
using BSON: @save, @load

true_data = npzread("data.npy")[1:20:4000,:]
data_len = size(true_data)[1]
x_data = true_data[:,1]
state_data = true_data[:,2:3]

u0 = [true_data[1,2],true_data[1,3]]
tspan = (0.0,true_data[data_len,1])
tsteps = 0.0:true_data[data_len,1]-true_data[data_len-1,1]:true_data[data_len,1]

my_nn = Chain(
  Dense(2, 32, relu),
  Dense(32, 2)) |> f32

p, re = Flux.destructure(my_nn)

function right_hand_side(du,u,p,t)
  m = re(p)
  nn_output = m(u)
  du[1] = nn_output[1]
  du[2] = nn_output[2]
end

prob = ODEProblem(right_hand_side,u0,tspan,p)
sol = solve(prob, Tsit5())

# Plot the solution
plot(sol)
savefig("Initial_NN_ODE.png")

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
train_loader = Flux.Data.DataLoader((transpose(state_data), tsteps), batchsize = k)
numEpochs = 200
losses=[]

test_batch = iterate(train_loader)
# print(check[1][2])
print(size(test_batch[1][1]))
print(size(test_batch[1][2]))
loss_ode(test_batch[1][1],test_batch[1][2])

cb() = begin
  
  l=loss_ode(transpose(state_data), tsteps)
  push!(losses, l)
  @show l

  pred = predict_ode(transpose(state_data),tsteps)
  
  plt = plot(x_data,hcat(transpose(pred),state_data), 
    line=(4, [:solid :solid :dash :dash]), ylim = (-6, 6),
    label=["y1-pred" "y2-pred" "y1-true" "y2-true"])
  
  display(plt)
  # Tell sciml_train to not halt the optimization. If return true, then
  # optimization stops.
  return false
end

opt=ADAM(0.01)
Flux.train!(loss_ode, Flux.params(p), ncycle(train_loader,numEpochs), opt, cb=Flux.throttle(cb, 10))

# my_nn_final = re(p)
# @save "mymodel.bson" my_nn_final

# # Inference
# @load "mymodel.bson" my_nn_final
# p, re = Flux.destructure(my_nn_final)


# prob_final = ODEProblem(right_hand_side, u0, tspan, p)
# sol_final = solve(prob_final,Tsit5())
# plot(sol_final)
# savefig("NN_ode_optimized.png") # Both state components close to 1