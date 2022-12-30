using DifferentialEquations, Flux, DiffEqFlux
using Plots

u0 = [1.0,1.0]
tspan = (0.0,10.0)
tsteps = 0.0:0.1:10.0

my_nn = Chain(
  Dense(2, 32, relu),
  Dense(32, 2)) |> f64

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

function predict_ode(p)
  tmp_prob = remake(prob, p = p)
  Array(solve(tmp_prob, Tsit5(), saveat = tsteps))
end

function loss_ode(p)
    pred = predict_ode(p)
    loss = sum(abs2, 1.0 .- pred)
    return loss, pred
end

callback = function (p, l, pred)
  display(l)
  plt = plot(pred, ylim = (0, 6))
  display(plt)
  # Tell sciml_train to not halt the optimization. If return true, then
  # optimization stops.
  return false
end

result_ode = DiffEqFlux.sciml_train(loss_ode, p,
                                    cb = callback,
                                    maxiters = 100)


prob_final = ODEProblem(right_hand_side, u0, tspan, result_ode)
sol_final = solve(prob_final,Tsit5())
plot(sol_final)
savefig("NN_ode_optimized.png") # Both state components close to 1